from unittest import TestCase
import numpy as np


class Cell:
    def __init__(self, i, j) -> None:
        self.i = i
        self.j = j
        self.empty = True
        self.cache_id = -1
        self.detached_cache_id = -1

    def alloc_cache_ids(self, cache_id_gen):
        assert self.empty
        self.empty = False
        self.cache_id = cache_id_gen()
        self.detached_cache_id = self.cache_id


class Table:
    def __init__(self, seq_len, next_cache_gen, detach_offset) -> None:
        self.seq_len = seq_len
        self.cells = [[None for j in range(seq_len)] for i in range(seq_len)]
        for i in range(seq_len):
            for j in range(seq_len):
                if i <= j:
                    self.cells[i][j] = Cell(i, j)
        self.active_cells = []
        self.next_cache_gen = next_cache_gen
        self.detach_offset = detach_offset
        self._start = -1
        self._end = -1

    @property
    def active_depth(self):
        return len(self.active_cells) - 1

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def prune(self, idx):
        start = self.seq_len
        end = 0
        self.active_cells[1][idx].detached_cache_id += self.detach_offset
        for layer_i, arr in enumerate(self.active_cells):
            for cpy_idx in range(max(0, idx - layer_i), min(idx + 1, len(arr))):
                if layer_i < len(self.active_cells) - 1 and cpy_idx < len(self.active_cells[layer_i + 1]):
                    arr[cpy_idx] = self.active_cells[layer_i + 1][cpy_idx]
                else:
                    if cpy_idx < len(arr) - 1:
                        # blank cells
                        start = min(start, cpy_idx)
                        end = max(end, cpy_idx)
                        next_i = arr[cpy_idx].i
                        next_j = arr[cpy_idx + 1].j
                        target = self.cells[next_i][next_j]
                        target.alloc_cache_ids(self.next_cache_gen)
                        arr[cpy_idx] = target
            if idx + 1 < len(arr):
                arr.pop(idx + 1)
            else:
                arr.pop(-1)
        self._start = start
        self._end = end + 1

    def expand_active_cells(self):
        arr = []
        depth = len(self.active_cells)
        for i in range(self.seq_len - depth):
            self.cells[i][i + depth].alloc_cache_ids(self.next_cache_gen)
            arr.append(self.cells[i][i + depth])
            if depth == 0:
                arr[-1].detached_cache_id += self.detach_offset
        self.active_cells.append(arr)
        self._start = 0
        self._end = len(arr)

    def finished(self, step):
        return step >= self.seq_len

    def gather_bilm(self, bos, eos, empty_cache_id):
        ids = []
        for pos in range(self.seq_len):
            if pos == 0:
                ids.append(bos)
            else:
                found = False
                for left_most in range(pos):
                    if not self.cells[left_most][pos - 1].empty:
                        found = True
                        ids.append(self.cells[left_most][pos - 1].cache_id)
                        break
                assert found
            if pos == self.seq_len - 1:
                ids.append(eos)
            else:
                found = False
                for right_most in range(self.seq_len - 1, pos, -1):
                    if not self.cells[pos + 1][right_most].empty:
                        found = True
                        ids.append(self.cells[pos + 1][right_most].cache_id)
                        break
                assert found
        assert len(ids) == 2 * self.seq_len
        return ids


class ChartTablesSimulator:
    def __init__(self, seq_lens, window_size, cache_id_offset, empty_cache_id, detach_offset) -> None:
        self.window_size = window_size
        self.current_step = 0
        self.last_prune_indices = None
        self.cache_id_offset = cache_id_offset
        self.emtpy_cache_id = empty_cache_id
        self.cache_id_front = cache_id_offset
        self.detach_offset = detach_offset
        self.tables = [Table(seq_len, self._next_cache_id, detach_offset)
                       for seq_len in seq_lens]

    def _next_cache_id(self):
        cache_id = self.cache_id_front
        self.cache_id_front += 1
        return cache_id

    def step(self):
        if self.current_step <= self.window_size:
            for table in self.tables:
                table.expand_active_cells()
        else:
            assert self.last_prune_indices is not None
            for table_i, table in enumerate(self.tables):
                if not table.finished(self.current_step):
                    table.prune(self.last_prune_indices[table_i])
            self.last_prune_indices = None
        cache_ids = []
        detached_ids = []
        group_size = min(self.current_step, self.window_size)
        for table in self.tables:
            if not table.finished(self.current_step):
                for pos in range(table.start, table.end):
                    for k in range(table.active_depth):
                        cache_ids.append(table.active_cells[k][pos].cache_id)
                        detached_ids.append(table.active_cells[k][pos].detached_cache_id)
                        pos_i = table.active_depth - 1 - k
                        pos_j = pos + 1 + k
                        cache_ids.append(
                            table.active_cells[pos_i][pos_j].cache_id)
                        detached_ids.append(
                            table.active_cells[pos_i][pos_j].detached_cache_id)
                        
        self.current_step += 1
        return cache_ids, detached_ids, group_size

    def gather_bilm(self, bos, eos, empty_id):
        ids = []
        for table in self.tables:
            ids.extend(table.gather_bilm(bos, eos, empty_id))
        return ids

    def prune(self, indices):
        self.last_prune_indices = indices


class TestCudaTableManager(TestCase):
    def test_cuda_tables(self):
        import torch
        import r2d2lib
        device = torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device('cuda')
        else:
            return
        window_size = 4

        test_count = 10
        test_batch_size = 8
        max_len = 128
        bos_cache_id = 0
        eos_cache_id = 1
        empty_cache_id = 2
        cache_offset = 3
        detach_offset = 2 * max_len * (window_size + 1) * test_batch_size + cache_offset
        max_group_size = window_size

        print(f'detach_offset: {detach_offset}')

        while True:  # test_count > 0:
            test_count -= 1
            tables = r2d2lib.TablesManager(False, window_size, 1)
            seq_lens = []
            if test_count % 2 == 0:
                for _ in range(test_batch_size):
                    seq_lens.append(np.random.randint(max_len - 1, max_len))
            else:
                rand_len = np.random.randint(max_len - 1, max_len)
                for _ in range(test_batch_size):
                    seq_lens.append(rand_len)
            enable_traj = np.random.randint(0, 2) == 1
            print(f'enable_traj: {enable_traj}')
            print(seq_lens)

            seq_len_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
            tables.encoding_start(seq_len_tensor, cache_offset, detach_offset, empty_cache_id)
            if enable_traj:
                merge_traj = []
                max_seq_len = max(seq_lens)
                for seq_len in seq_lens:
                    merge_order = [_ for _ in range(seq_len - 1)]
                    np.random.shuffle(merge_order)
                    merge_order.extend([_ for _ in range(seq_len, max_seq_len)])
                    merge_traj.append(merge_order)
                merge_traj = torch.tensor(merge_traj, device=device)
                merge_traj_np = merge_traj.to('cpu').data.numpy()
                tables.set_merge_trajectories(merge_traj)

            pytables = ChartTablesSimulator(seq_lens, window_size, cache_offset,
                                            empty_cache_id, detach_offset)
            # cache_id tensor
            cache_ids = torch.full([sum(seq_lens) * max_group_size * 2],
                                   0, dtype=torch.int, requires_grad=False, device=device)
            log_p_ids = torch.full([sum(seq_lens) * max_group_size * 2],
                                   0,
                                   dtype=torch.int,
                                   requires_grad=False,
                                   device=device)
            bigram_scores = torch.full(
                [len(seq_lens), max(seq_lens)],
                0.0,
                dtype=torch.float,
                requires_grad=False,
                device=device)
            span_lens = torch.full([sum(seq_lens)],
                                   1.0,
                                   dtype=torch.float,
                                   requires_grad=False,
                                   device=device)
            batch_size, current_size, _, _, _ = tables.step(cache_ids, log_p_ids,
                                                                 span_lens, bigram_scores,
                                                                 torch.zeros_like(bigram_scores))
            log_p_batch = torch.full([batch_size, 1],
                                     0.0,
                                     dtype=torch.float,
                                     device=device)
            tables.step_over(log_p_batch, log_p_batch)
            pytables.step()

            prune_step = 0
            while not tables.finished():
                bigram_scores.fill_(float("-inf"))  # (batch_size, max_len)
                noise = torch.zeros_like(bigram_scores, requires_grad=False)
                span_lens = torch.ones_like(span_lens,
                                            dtype=torch.float,
                                            requires_grad=False)
                cache_ids.fill_(empty_cache_id)
                log_p_ids.fill_(detach_offset)
                # start = time.time()
                noise = -torch.empty_like(bigram_scores,
                                          memory_format=torch.legacy_contiguous_format,
                                          requires_grad=False).exponential_().log()
                batch_size, current_size, group_size_a, _, _ = tables.step(
                    cache_ids, log_p_ids, span_lens, bigram_scores, noise)

                prune_indices = None
                if enable_traj:
                    if pytables.current_step > window_size:
                        prune_indices = merge_traj_np[:, prune_step]
                        for batch_i, indices in enumerate(merge_traj_np):
                            for idx in range(len(indices)):
                                if merge_traj_np[batch_i, idx] > prune_indices[batch_i]:
                                    merge_traj_np[batch_i, idx] -= 1
                        prune_step += 1
                else:
                    prune_indices = (bigram_scores + noise).argmax(dim=-1)  # [0] * test_batch_size  # 
                # print(f'prune indices: {prune_indices}')
                pytables.prune(prune_indices)

                sim_cache_ids, sim_detach_ids, group_size_b = pytables.step()

                self.assertEqual(group_size_a, group_size_b, f'current step: {tables.current_step()}')
                self.assertTrue(batch_size > 0)
                sim_cache_ids = torch.tensor(sim_cache_ids, device=device)
                sim_cache_ids = sim_cache_ids.view(-1, group_size_b, 2)
                sim_detach_ids = torch.tensor(sim_detach_ids, device=device)
                sim_detach_ids = sim_detach_ids.view(-1, group_size_b, 2)

                ids = cache_ids[:batch_size * group_size_a * 2]
                ids = ids.view(-1, group_size_a, 2)
                log_p_ids_ = log_p_ids[:batch_size * group_size_a * 2]
                log_p_ids_ = log_p_ids_.view(-1, group_size_a, 2)
                self.assertTrue(ids.shape == sim_cache_ids.shape, f'{ids.shape} vs {sim_cache_ids.shape}')
                self.assertTrue(torch.all(sim_cache_ids == ids), f'{sim_cache_ids} vs {ids}')
                self.assertTrue(log_p_ids_.shape == sim_detach_ids.shape,
                                f'{log_p_ids_.shape} vs {sim_cache_ids.shape}')
                self.assertTrue(torch.all(sim_detach_ids == log_p_ids_), f'{sim_detach_ids} vs {log_p_ids_}')

                # indices_selected: (total_size, 1)
                tables.beam_select(torch.randint(0, group_size_b, [batch_size, current_size],
                                                 dtype=torch.int, device=device))

                # ASSUME: log_p_ij_step (batch_size, 1)
                # candidats_log_p: (batch_size, depth)
                tables.step_over(torch.rand([batch_size, current_size], device=device),
                                 torch.rand([batch_size, min(tables.current_step(), window_size) *
                                             current_size * current_size], device=device))
            bilm_ids_a = pytables.gather_bilm(bos_cache_id, eos_cache_id, empty_cache_id)
            bilm_ids_b = torch.full([sum(seq_lens) * 2], 0, dtype=torch.int, device=device)
            tables.prepare_bilm(bilm_ids_b, bos_cache_id, eos_cache_id)
            bilm_ids_a = torch.tensor(bilm_ids_a, dtype=torch.int, device=device)
            bilm_ids_a = bilm_ids_a.view(-1, 1, 2)
            bilm_ids_b = bilm_ids_b.view(-1, 1, 2)
            self.assertTrue(torch.all(bilm_ids_a == bilm_ids_b), f'{bilm_ids_a} vs {bilm_ids_b}')

            tables.encoding_over()

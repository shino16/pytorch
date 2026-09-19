# Owner(s): ["oncall: distributed"]

import sys
import threading
import unittest

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10dBackendTest,
    CUDA_BACKENDS,
    instantiate_backend_tests,
)

from torch.distributed._functional_collectives import (
    _allow_collective_wait_outside_capture,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_WITH_ROCM,
    TestCase,
)


ASYNC_OPS = (False, True)


class CollectiveCaptureScopeTest(TestCase):
    def test_scope_restores_state(self):
        set_enabled = (
            torch._C._distributed_c10d._set_allow_collective_wait_outside_capture
        )
        previous = set_enabled(False)
        try:
            with _allow_collective_wait_outside_capture():
                with self.assertRaisesRegex(RuntimeError, "scope exception"):
                    with _allow_collective_wait_outside_capture():
                        raise RuntimeError("scope exception")
                self.assertTrue(set_enabled(True))
                other_thread = []
                thread = threading.Thread(
                    target=lambda: other_thread.append(set_enabled(False))
                )
                thread.start()
                thread.join()
                self.assertEqual(other_thread, [False])
            self.assertFalse(set_enabled(False))
        finally:
            set_enabled(previous)


class AbstractCUDAGraphsTest(C10dBackendTest):
    def _tensor(self, dtype, rank=None):
        rank = self.rank if rank is None else rank
        return torch.full((4,), rank + 1, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _capture_and_replay(self, op, async_op):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            work = op()
            if async_op:
                self.assertIsNotNone(work)
                work.wait()
            else:
                self.assertIsNone(work)
        graph.replay()
        torch.cuda.synchronize()

    def _test_all_reduce(self, dtype, async_op):
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.all_reduce(tensor, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)

    def _test_broadcast(self, dtype, async_op):
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.broadcast(tensor, src=0, async_op=async_op),
            async_op,
        )
        self.assertEqual(tensor, self._tensor(dtype, rank=0))

    def _test_all_gather(self, dtype, async_op):
        tensor = self._tensor(dtype)
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        self._capture_and_replay(
            lambda: dist.all_gather(output, tensor, async_op=async_op),
            async_op,
        )
        for rank, result in enumerate(output):
            self.assertEqual(result, self._tensor(dtype, rank))

    def _test_all_gather_single(self, dtype, async_op):
        tensor = self._tensor(dtype)
        output = torch.empty(
            self.world_size * tensor.numel(),
            dtype=dtype,
            device=self.device,
        )
        self._capture_and_replay(
            lambda: dist.all_gather_single(output, tensor, async_op=async_op),
            async_op,
        )
        expected = torch.cat(
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
        )
        self.assertEqual(output, expected)

    def _test_reduce(self, dtype, async_op):
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.reduce(tensor, dst=0, async_op=async_op),
            async_op,
        )
        if self.rank == 0:
            expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
            self.assertEqual(tensor, expected)

    def _test_gather(self, dtype, async_op):
        tensor = self._tensor(dtype)
        output = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        self._capture_and_replay(
            lambda: dist.gather(tensor, output, dst=0, async_op=async_op),
            async_op,
        )
        if self.rank == 0:
            for rank, result in enumerate(output):
                self.assertEqual(result, self._tensor(dtype, rank))

    def _test_scatter(self, dtype, async_op):
        output = torch.empty(4, dtype=dtype, device=self.device)
        inputs = (
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
            if self.rank == 0
            else None
        )
        self._capture_and_replay(
            lambda: dist.scatter(output, inputs, src=0, async_op=async_op),
            async_op,
        )
        self.assertEqual(output, self._tensor(dtype, self.rank))

    def _test_reduce_scatter(self, dtype, async_op):
        inputs = [self._tensor(dtype) for _ in range(self.world_size)]
        output = torch.empty_like(inputs[0])
        self._capture_and_replay(
            lambda: dist.reduce_scatter(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(output, sum(range(1, self.world_size + 1)))
        self.assertEqual(output, expected)

    def _test_reduce_scatter_single(self, dtype, async_op):
        inputs = torch.cat([self._tensor(dtype) for _ in range(self.world_size)])
        output = torch.empty(4, dtype=dtype, device=self.device)
        self._capture_and_replay(
            lambda: dist.reduce_scatter_single(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(output, sum(range(1, self.world_size + 1)))
        self.assertEqual(output, expected)

    def _test_all_to_all(self, dtype, async_op):
        inputs = [self._tensor(dtype, self.rank) for _ in range(self.world_size)]
        outputs = [torch.empty_like(inputs[0]) for _ in range(self.world_size)]
        self._capture_and_replay(
            lambda: dist.all_to_all(outputs, inputs, async_op=async_op),
            async_op,
        )
        for rank, result in enumerate(outputs):
            self.assertEqual(result, self._tensor(dtype, rank))

    def _test_all_to_all_single(self, dtype, async_op):
        inputs = torch.cat(
            [self._tensor(dtype, self.rank) for _ in range(self.world_size)]
        )
        output = torch.empty_like(inputs)
        self._capture_and_replay(
            lambda: dist.all_to_all_single(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.cat(
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
        )
        self.assertEqual(output, expected)

    def test_collectives(self):
        self._init_pg()
        warmup = torch.ones(1, device=self.device)
        dist.all_reduce(warmup)
        torch.cuda.synchronize()

        tests = (
            self._test_all_reduce,
            self._test_broadcast,
            self._test_all_gather,
            self._test_all_gather_single,
            self._test_reduce,
            self._test_gather,
            self._test_scatter,
            self._test_reduce_scatter,
            self._test_reduce_scatter_single,
            self._test_all_to_all,
            self._test_all_to_all_single,
        )
        for test in tests:
            for dtype in self.dtypes:
                for async_op in ASYNC_OPS:
                    with self.subTest(
                        collective=test.__name__,
                        dtype=dtype,
                        async_op=async_op,
                    ):
                        test(dtype, async_op)

    def test_barrier(self):
        if not self.supports_cuda_graph_barrier:
            self.skipTest(f"{self.backend_name} barrier does not support CUDA graphs")
        self._init_pg()
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        for async_op in ASYNC_OPS:
            self._capture_and_replay(
                lambda: dist.barrier(
                    device_ids=[self.rank],
                    async_op=async_op,
                ),
                async_op,
            )

    def test_complex_collectives(self):
        self._init_pg()
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        for dtype in self.complex_dtypes:
            for async_op in ASYNC_OPS:
                for test in (
                    self._test_all_reduce,
                    self._test_broadcast,
                    self._test_reduce,
                ):
                    with self.subTest(
                        collective=test.__name__,
                        dtype=dtype,
                        async_op=async_op,
                    ):
                        test(dtype, async_op)


instantiate_backend_tests(
    globals(), "CUDAGraphs", AbstractCUDAGraphsTest, CUDA_BACKENDS
)


@unittest.skipIf(TEST_WITH_ROCM, "CUDA external event capture")
@unittest.skipIf(not dist.is_nccl_available(), "NCCL required")
class NCCLCrossCaptureTest(C10dBackendTest, MultiProcessTestCase):
    backend_name = "nccl-legacy"

    @skip_if_lt_x_gpu(2)
    def test_selected_event_nodes(self, device):
        from torch.cuda._utils import _check_cuda_bindings as check, _HAS_CUDA_BINDINGS

        if not _HAS_CUDA_BINDINGS:
            self.skipTest("cuda.bindings required to inspect event nodes")
        from cuda.bindings import runtime

        self._init_pg()
        inp = torch.ones(16, device=self.device)
        dist.all_reduce(inp)
        torch.cuda.synchronize()

        def capture(selected):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                dist.all_reduce(inp, async_op=True).wait()
                if selected:
                    with _allow_collective_wait_outside_capture():
                        work = dist.all_reduce(inp, async_op=True)
                else:
                    work = dist.all_reduce(inp, async_op=True)
                work.wait()
            _, count = check(runtime.cudaGraphGetNodes(graph.raw_cuda_graph()))
            nodes, _ = check(
                runtime.cudaGraphGetNodes(graph.raw_cuda_graph(), numNodes=count)
            )
            records = sum(
                check(runtime.cudaGraphNodeGetType(node))
                == runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord
                for node in nodes[:count]
            )
            return graph, records

        ordinary, ordinary_records = capture(False)
        selected, selected_records = capture(True)
        self.assertEqual(selected_records, ordinary_records + 1)
        for graph in (ordinary, selected):
            inp.fill_(1)
            graph.replay()
            self.assertEqual(inp, torch.full_like(inp, self.world_size**2))

    @skip_if_lt_x_gpu(2)
    @parametrize("graph_partition", [False, True])
    @parametrize("out_variant", [False, True])
    def test_compiled_selection(self, device, graph_partition, out_variant):
        from torch._inductor.utils import run_and_get_code

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(inp):
            if out_variant:
                local = torch.empty(self.world_size * inp.numel(), device=inp.device)
                torch.ops._c10d_functional.all_gather_into_tensor_out(
                    inp, self.world_size, group_name, out=local
                )
            else:
                local = torch.ops._c10d_functional.all_reduce(inp, "sum", group_name)
            local = torch.ops._c10d_functional.wait_tensor(local.view(-1, 4))
            pending = torch.ops._c10d_functional.all_reduce(inp + 1, "sum", group_name)
            return local + 1, pending

        torch._dynamo.reset()
        try:
            compiled = torch.compile(
                model,
                fullgraph=True,
                options={"triton.cudagraphs": True, "graph_partition": graph_partition},
            )
            inp = torch.ones(16, device=self.device)
            (_, pending), code = run_and_get_code(compiled, inp)
            torch.ops._c10d_functional.wait_tensor(pending)
            self.assertEqual(
                "\n".join(code).count("_allow_collective_wait_outside_capture():"), 1
            )
            for offset in range(5):
                inp.fill_(self.rank + offset + 1)
                local, pending = compiled(inp)
                waited = torch.ops._c10d_functional.wait_tensor(pending)
                total = sum(rank + offset + 1 for rank in range(self.world_size))
                if out_variant:
                    expected = torch.cat(
                        [
                            torch.full_like(inp, rank + offset + 2)
                            for rank in range(self.world_size)
                        ]
                    ).view(-1, 4)
                else:
                    expected = torch.full_like(local, total + 1)
                self.assertEqual(local, expected)
                self.assertEqual(
                    waited, torch.full_like(waited, total + self.world_size)
                )
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    def test_compiled_mixed_static_dynamic_partitions(self, device):
        import ast

        from torch._inductor.utils import run_and_get_code

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(inp, other, dynamic):
            pending = torch.ops._c10d_functional.all_gather_into_tensor(
                inp, self.world_size, group_name
            )
            ready = torch.ops._c10d_functional.all_reduce(other, "sum", group_name)
            ready = torch.ops._c10d_functional.wait_tensor(ready)
            middle = torch.ops._c10d_functional.all_reduce(
                dynamic + ready[0], "sum", group_name
            )
            middle = torch.ops._c10d_functional.wait_tensor(middle)
            # The data dependency prevents hoisting dynamic computation before
            # the first partition. Collective ordering keeps the second gather
            # after it, while the final wait operands retain static shapes.
            local = torch.ops._c10d_functional.all_gather_into_tensor(
                other, self.world_size, group_name
            )
            pending, local = torch.ops._c10d_functional.wait_tensors([pending, local])
            return pending + local + middle.sum()

        torch._dynamo.reset()
        try:
            compiled = torch.compile(
                model,
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    "graph_partition": True,
                    "triton.cudagraph_skip_dynamic_graphs": True,
                },
            )
            inp = torch.ones(16, device=self.device)
            other = torch.full_like(inp, 2)
            dynamic = torch.ones(8, device=self.device)
            torch._dynamo.mark_dynamic(dynamic, 0)
            _, code = run_and_get_code(compiled, inp, other, dynamic)
            source = "\n".join(code)
            partitions = {
                node.name: ast.get_source_segment(source, node)
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef)
                and node.name.startswith("partition_")
            }
            self.assertEqual(set(partitions), {"partition_0", "partition_1"})
            scope = "_allow_collective_wait_outside_capture():"
            self.assertEqual(source.count(scope), 1)
            self.assertIn(scope, partitions["partition_0"])
            self.assertIn("all_gather_into_tensor", partitions["partition_0"])
            self.assertIn("all_gather_into_tensor", partitions["partition_1"])
            self.assertIn("wait_tensors.default", partitions["partition_1"])
            # The dynamic reduction runs between the two captured partitions.
            self.assertRegex(
                source,
                r"(?s)self\.partitions\[0\].*\.run\(.*self\.partitions\[1\]",
            )
            with torch.compiler.set_stance("fail_on_recompile"):
                for offset, size in enumerate((8, 8, 16, 16, 8, 8)):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp.fill_(self.rank + offset + 1)
                    other.fill_(self.rank + offset + 2)
                    dynamic = torch.full((size,), float(offset + 1), device=self.device)
                    torch._dynamo.mark_dynamic(dynamic, 0)
                    output = compiled(inp, other, dynamic)
                    expected = torch.cat(
                        [
                            torch.full_like(
                                inp,
                                2 * (rank + offset)
                                + 3
                                + self.world_size
                                * size
                                * (
                                    offset
                                    + 1
                                    + self.world_size * (offset + 2)
                                    + self.world_size * (self.world_size - 1) / 2
                                ),
                            )
                            for rank in range(self.world_size)
                        ]
                    )
                    self.assertEqual(output, expected)
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    @parametrize("eager_op", ["all_reduce_", "wait_tensor"])
    def test_compiled_partition_boundary(self, device, eager_op):
        from torch._inductor.utils import run_and_get_code

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(inp):
            pending = torch.ops._c10d_functional.all_reduce(inp, "sum", group_name)
            if eager_op == "all_reduce_":
                return pending
            return torch.ops._c10d_functional.wait_tensor(pending) + 1

        torch._dynamo.reset()
        try:
            compiled = torch.compile(
                model,
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    "graph_partition": True,
                    "custom_should_partition_ops": [f"_c10d_functional::{eager_op}"],
                },
            )
            inp = torch.ones(16, device=self.device)
            output, code = run_and_get_code(compiled, inp)
            if eager_op == "all_reduce_":
                torch.ops._c10d_functional.wait_tensor(output)
            self.assertEqual(
                "\n".join(code).count("_allow_collective_wait_outside_capture():"),
                int(eager_op == "wait_tensor"),
            )
            for offset in range(5):
                inp.fill_(self.rank + offset + 1)
                expected = sum(rank + offset + 1 for rank in range(self.world_size))
                output = compiled(inp)
                if eager_op == "all_reduce_":
                    output = torch.ops._c10d_functional.wait_tensor(output) + 1
                self.assertEqual(output, torch.full_like(inp, expected + 1))
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    def test_synchronous_shard_alltoall(self, device):
        from torch._inductor.utils import run_and_get_code

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(inp):
            return torch.ops._dtensor.shard_dim_alltoall(inp, 0, 1, group_name)

        torch._dynamo.reset()
        try:
            compiled = torch.compile(
                model,
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    "graph_partition": False,
                    "decompose_shard_dim_alltoall": False,
                },
            )
            inp = torch.ones(4, 4, device=self.device)
            _, code = run_and_get_code(compiled, inp)
            self.assertNotIn(
                "_allow_collective_wait_outside_capture():", "\n".join(code)
            )
            for offset in range(5):
                inp.fill_(self.rank + offset + 1)
                expected = torch.cat(
                    [
                        torch.full(
                            (4, 2),
                            rank + offset + 1,
                            device=self.device,
                            dtype=inp.dtype,
                        )
                        for rank in range(self.world_size)
                    ]
                )
                self.assertEqual(compiled(inp), expected)
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    def test_escaping_fx_wrapper_rejected(self, device):
        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        def model(inp):
            return torch.ops._c10d_functional.all_reduce(inp, "sum", group_name)

        torch._dynamo.reset()
        try:
            compiled = torch.compile(
                model,
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    "graph_partition": False,
                    "fx_wrapper": True,
                },
            )
            with self.assertRaisesRegex(
                torch._dynamo.exc.BackendCompilerFailed,
                "Cross-capture collective waits require a top-level",
            ):
                compiled(torch.ones(16, device=self.device))
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    def test_selected_coalesced(self, device):
        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name
        inp = torch.ones(16, device=self.device)
        dist.all_reduce(inp)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), _allow_collective_wait_outside_capture():
            pending = torch.ops._c10d_functional.all_reduce_coalesced(
                [inp, inp + 1], "sum", group_name
            )
        consumer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(consumer):
            outputs = torch.ops._c10d_functional.wait_tensors(pending)
        for offset in range(5):
            inp.fill_(self.rank + offset + 1)
            graph.replay()
            consumer.replay()
            for i, output in enumerate(outputs):
                total = sum(rank + offset + i + 1 for rank in range(self.world_size))
                self.assertEqual(output, torch.full_like(output, total))

    @skip_if_lt_x_gpu(2)
    @parametrize("owner", ["producer", "consumer"])
    def test_event_lifetime(self, device, owner):
        from torch.cuda._utils import _check_cuda_bindings as check, _HAS_CUDA_BINDINGS

        if not _HAS_CUDA_BINDINGS:
            self.skipTest("cuda.bindings required to inspect event ownership")
        from cuda.bindings import runtime

        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name
        inp = torch.ones(16, device=self.device)
        dist.all_reduce(inp)
        torch.cuda.synchronize()

        def capture(wait_inside):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph), _allow_collective_wait_outside_capture():
                pending = torch.ops._c10d_functional.all_reduce(inp, "sum", group_name)
                if wait_inside:
                    torch.ops._c10d_functional.wait_tensor(pending)
            return graph, pending

        def recorded_events(graph):
            raw = graph.raw_cuda_graph()
            _, count = check(runtime.cudaGraphGetNodes(raw))
            nodes, _ = check(runtime.cudaGraphGetNodes(raw, numNodes=count))
            return {
                int(check(runtime.cudaGraphEventRecordNodeGetEvent(node)))
                for node in nodes[:count]
                if check(runtime.cudaGraphNodeGetType(node))
                == runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord
            }

        reference, _ = capture(True)
        shared_events = recorded_events(reference)

        producer, pending = capture(owner == "producer")
        # Exclude NCCL serialization events shared with the reference graph.
        events = recorded_events(producer) - shared_events
        self.assertEqual(len(events), 1)
        if owner == "consumer":
            consumer = torch.cuda.CUDAGraph()
            with torch.cuda.graph(consumer):
                torch.ops._c10d_functional.wait_tensor(pending)
            producer.reset()
        del pending

        # Keep the churn graphs live so each event must have a distinct owner.
        graphs = []
        for _ in range(32):
            graph, pending = capture(True)
            self.assertTrue(events.isdisjoint(recorded_events(graph)))
            graphs.append(graph)
            del pending

    @skip_if_lt_x_gpu(2)
    def test_compiled_graph_break(self, device):
        from torch._dynamo.utils import counters
        from torch.testing._internal.inductor_utils import HAS_GPU

        if not HAS_GPU:
            self.skipTest("Inductor GPU compilation requires Triton")
        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name

        @torch.compile(mode="reduce-overhead")
        def model(inp):
            pending = torch.ops._c10d_functional.all_reduce(inp, "sum", group_name)
            torch._dynamo.graph_break()
            waited = torch.ops._c10d_functional.wait_tensor(pending)
            return waited.view(4, 4)[:, :2] + 1

        torch._dynamo.reset()
        skips_before = counters["inductor"]["cudagraph_skips"]
        try:
            for offset in range(10):
                inp = torch.full((16,), self.rank + offset + 1.0, device=self.device)
                expected = sum(rank + offset + 1.0 for rank in range(self.world_size))
                self.assertEqual(
                    model(inp), torch.full((4, 2), expected + 1, device=self.device)
                )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], skips_before)
        finally:
            torch._dynamo.reset()

    @skip_if_lt_x_gpu(2)
    @parametrize("wait_location", ["same_graph", "other_graph", "eager"])
    @parametrize("collective", ["all_reduce", "all_gather_into_tensor"])
    def test_wait_tensor(self, device, wait_location, collective):
        self._init_pg()
        group_name = dist.distributed_c10d._get_default_group().group_name
        inp = torch.ones(16, device=self.device)
        dist.all_reduce(inp)
        torch.cuda.synchronize()

        def launch(tensor):
            if collective == "all_reduce":
                return torch.ops._c10d_functional.all_reduce(tensor, "sum", group_name)
            return torch.ops._c10d_functional.all_gather_into_tensor(
                tensor, self.world_size, group_name
            )

        def wait():
            return [
                torch.ops._c10d_functional.wait_tensor(tensor.view(-1, 4))[:, :2] + 1
                for tensor in pending
            ]

        producer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(producer), _allow_collective_wait_outside_capture():
            pending = [launch(inp), launch(inp + 1)]
            if wait_location == "same_graph":
                outputs = wait()
        consumer = None
        if wait_location == "other_graph":
            consumer = torch.cuda.CUDAGraph()
            with torch.cuda.graph(consumer):
                outputs = wait()

        replay_stream = torch.cuda.Stream()
        for offset in range(10):
            inp.fill_(self.rank + offset + 1)
            if consumer is not None:
                replay_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(replay_stream):
                    producer.replay()
                consumer.replay()
            else:
                producer.replay()
                if wait_location == "eager":
                    outputs = wait()
            for index, output in enumerate(outputs):
                if collective == "all_reduce":
                    value = sum(
                        rank + offset + index + 1 for rank in range(self.world_size)
                    )
                    expected = torch.full_like(output, value + 1)
                else:
                    expected = torch.cat(
                        [
                            torch.full(
                                (4, 2),
                                rank + offset + index + 2,
                                device=self.device,
                                dtype=inp.dtype,
                            )
                            for rank in range(self.world_size)
                        ]
                    )
                self.assertEqual(output, expected)


instantiate_device_type_tests(NCCLCrossCaptureTest, globals(), only_for="cuda")


@unittest.skipIf(not TEST_WITH_ROCM, "ROCm capture restriction")
@unittest.skipIf(not dist.is_nccl_available(), "NCCL required")
class NCCLCaptureWaitROCmTest(C10dBackendTest, MultiProcessTestCase):
    backend_name = "nccl-legacy"

    @skip_if_lt_x_gpu(2)
    @parametrize("wait_in_capture", [False, True])
    def test_wait_outside_original_capture(self, device, wait_in_capture):
        self._init_pg()
        inp = torch.ones(4, device=self.device)
        dist.all_reduce(inp, async_op=True).wait()
        inp.fill_(1)
        torch.cuda.synchronize()

        producer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(producer), _allow_collective_wait_outside_capture():
            work = dist.all_reduce(inp, async_op=True)
            work.wait()
        producer.replay()
        torch.cuda.synchronize()
        self.assertEqual(inp, torch.full_like(inp, self.world_size))

        message = "outside its original capture is not supported on ROCm"
        if wait_in_capture:
            consumer = torch.cuda.CUDAGraph()
            with torch.cuda.graph(consumer):
                with self.assertRaisesRegex(NotImplementedError, message):
                    work.wait()
        else:
            with self.assertRaisesRegex(NotImplementedError, message):
                work.wait()


instantiate_device_type_tests(NCCLCaptureWaitROCmTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()

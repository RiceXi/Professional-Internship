import unittest

from scripts.gpu_matrix import classify


class MatrixClassificationTests(unittest.TestCase):
    def test_capacity_boundary_does_not_hide_cuda_errors(self):
        report = {
            'status': 'failed',
            'reason': 'RuntimeError: parent completed or failed before fork',
            'storage_roundtrip': {'status': 'passed'},
            'cases': {'fixed_swap': {'requests': [
                {'status': 'capacity_rejected', 'error': 'request attention working set exceeds physical KV capacity'}]}},
        }
        self.assertEqual(classify(report, 1, True), 'capacity_boundary')
        self.assertEqual(classify(report, 1, False), 'failed')
        report['reason'] = 'OutOfMemoryError: CUDA out of memory'
        self.assertEqual(classify(report, 1, True), 'failed')
        report['reason'] = 'RuntimeError: parent completed or failed before fork'
        report['cases']['fixed_swap']['requests'][0]['error'] = 'host page pool exhausted'
        self.assertEqual(classify(report, 1, True), 'failed')

    def test_exit_code_and_report_must_agree(self):
        self.assertEqual(classify({'status': 'passed'}, 0, False), 'passed')
        self.assertEqual(classify({'status': 'passed'}, 1, False), 'failed')
        self.assertEqual(classify({'status': 'unavailable'}, 2, False), 'unavailable')
        self.assertEqual(classify({}, 0, False), 'failed')

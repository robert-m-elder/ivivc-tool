"""Profiling-tool checks runnable without Flask: python -m unittest discover
-s tests -p test_profile_app.py
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import pstats
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from tools.profile_app import ACTIVE, ProfileRequests, Timings, install_timers, timed


class TimerTests(unittest.TestCase):
    def test_existing_import_aliases_share_one_timer(self):
        source = ModuleType('utilities.profile_fixture')
        alias = ModuleType('ivivc_core.profile_fixture')
        result = object()
        source.work = lambda: result
        alias.work = source.work
        with patch.dict(sys.modules, {source.__name__: source, alias.__name__: alias}), \
                patch('tools.profile_app.TARGETS', {source.__name__: ('work',)}):
            install_timers()
            self.assertIs(source.work, alias.work)
            timings = Timings()
            token = ACTIVE.set(timings)
            try:
                self.assertIs(alias.work(), result)
            finally:
                ACTIVE.reset(token)
            self.assertEqual(timings.records()[0]['calls'], 1)

    def test_nested_calls_preserve_values_and_exclude_child_time(self):
        result = object()
        inner = timed(lambda: result, 'inner')
        outer = timed(lambda **kwargs: inner(), 'outer')
        timings = Timings()
        token = ACTIVE.set(timings)
        try:
            with patch('tools.profile_app.perf_counter', side_effect=[0., 2., 5., 10.]):
                self.assertIs(outer(model_name='exponential'), result)
        finally:
            ACTIVE.reset(token)
        rows = {row['path']: row for row in timings.records()}
        self.assertEqual(rows['outer']['inclusive_seconds'], 10.)
        self.assertEqual(rows['outer']['exclusive_seconds'], 7.)
        self.assertEqual(rows['outer > inner']['model'], 'exponential')
        self.assertEqual(rows['outer > inner']['inclusive_seconds'], 3.)

    def test_exception_identity_and_context_are_preserved(self):
        exception = ValueError('expected')
        def fail():
            raise exception
        timings = Timings()
        token = ACTIVE.set(timings)
        try:
            with self.assertRaises(ValueError) as caught:
                timed(fail, 'failure')()
            self.assertIs(caught.exception, exception)
            self.assertEqual(timings.stack, [])
            self.assertEqual(timings.records()[0]['exceptions'], 1)
        finally:
            ACTIVE.reset(token)

    def test_plot_model_is_identified_beneath_unlabelled_parent(self):
        def curve():
            pass
        curve.__module__ = 'models.exponential'
        inner = timed(lambda model_info, **kwargs: None, 'plot')
        outer = timed(lambda: inner({'model_function': curve}, prediction_mode='tau_ratio'), 'request')
        timings = Timings()
        token = ACTIVE.set(timings)
        try:
            outer()
        finally:
            ACTIVE.reset(token)
        row = next(row for row in timings.records() if row['path'].endswith('plot[tau_ratio]'))
        self.assertEqual(row['model'], 'exponential')


class RequestTests(unittest.TestCase):
    def test_timing_and_detailed_files_preserve_response_and_close_it(self):
        for detailed in (False, True):
            with self.subTest(detailed=detailed), tempfile.TemporaryDirectory() as directory:
                closed = []
                def body():
                    try:
                        yield b'hello'
                        yield b' world'
                    finally:
                        closed.append(True)
                def application(environ, start_response):
                    timed(lambda **kwargs: None, 'analysis')(model_name='exponential')
                    start_response('200 OK', [('Content-Type', 'text/plain')])
                    return body()
                headers = []
                middleware = ProfileRequests(application, directory, detailed=detailed, label='test')
                with patch('sys.stdout', new=io.StringIO()):
                    response = middleware({'REQUEST_METHOD': 'POST', 'PATH_INFO': '/'},
                                          lambda status, values, exc_info=None: headers.extend(values))
                    self.assertEqual(b''.join(response), b'hello world')
                self.assertEqual(closed, [True])
                self.assertIsNone(ACTIVE.get())
                self.assertTrue(any(name == 'Server-Timing' for name, value in headers))
                files = list(Path(directory).glob('*.json'))
                self.assertEqual(len(files), 1)
                metadata = json.loads(files[0].read_text())
                self.assertEqual(metadata['response_bytes'], 11)
                self.assertEqual(metadata['status'], '200 OK')
                self.assertEqual(metadata['label'], 'test')
                with files[0].with_suffix('.csv').open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(rows[0]['model'], 'exponential')
                if detailed:
                    self.assertTrue(pstats.Stats(str(files[0].with_suffix('.prof'))).stats)
                    self.assertIn('Suspect functions', files[0].with_suffix('.txt').read_text())
                else:
                    self.assertFalse(files[0].with_suffix('.prof').exists())

    def test_get_is_unmodified_and_saving_failure_does_not_change_post(self):
        def application(environ, start_response):
            start_response('200 OK', [])
            return [b'ok']
        with tempfile.TemporaryDirectory() as directory:
            middleware = ProfileRequests(application, directory)
            self.assertEqual(middleware({'REQUEST_METHOD': 'GET', 'PATH_INFO': '/'},
                                        lambda *args: None), [b'ok'])
            self.assertEqual(list(Path(directory).iterdir()), [])
            with patch('tools.profile_app.save_profile', side_effect=OSError('disk full')), \
                    patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
                response = middleware({'REQUEST_METHOD': 'POST', 'PATH_INFO': '/'}, lambda *args: None)
                self.assertEqual(b''.join(response), b'ok')
            self.assertIsNone(ACTIVE.get())

    def test_application_exception_saves_diagnostics_and_cleans_context(self):
        exception = RuntimeError('expected')
        def application(environ, start_response):
            raise exception
        with tempfile.TemporaryDirectory() as directory, patch('sys.stdout', new=io.StringIO()):
            middleware = ProfileRequests(application, directory, detailed=True)
            with self.assertRaises(RuntimeError) as caught:
                list(middleware({'REQUEST_METHOD': 'POST', 'PATH_INFO': '/'}, lambda *args: None))
            self.assertIs(caught.exception, exception)
            self.assertIsNone(ACTIVE.get())
            metadata = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(metadata['status'], 'exception')


if __name__ == '__main__':
    unittest.main()

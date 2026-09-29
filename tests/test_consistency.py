import copy
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from consistency_suite import build_scenarios, finalize, summarize, percentile
from visual_tests import render


class ConsistencyTests(unittest.TestCase):
    def case(self):
        return dict(focus='a', expected='a', trials=[])

    def trial(self, repeat, choice='a', p=.7, error=None):
        return dict(repeat=repeat, choice=choice, probabilities={'a':p, 'b':1-p},
                    elapsed_seconds=.1, error=error)

    def test_identical_repeats_are_stable_not_independent_accuracy_evidence(self):
        case = self.case()
        case['trials'] = [self.trial(i) for i in range(1,21)]
        summary = summarize(case)
        self.assertEqual(summary['agreement'],1)
        self.assertEqual(summary['flip_rate'],0)
        self.assertEqual(summary['p_min'],summary['p_max'])
        self.assertEqual(summary['median_ms'],100)

    def test_error_breaks_adjacency_and_is_not_a_wrong_choice(self):
        case = self.case()
        case['trials'] = [self.trial(1),self.trial(2,error='missing'),self.trial(3,'b',.2)]
        result = summarize(case)
        self.assertEqual(result['errors'],1)
        self.assertEqual(result['accuracy'],.5)
        self.assertIsNone(result['flip_rate'])

    def test_known_choice_flip(self):
        case=self.case()
        case['trials']=[self.trial(1),self.trial(2,'b',.2),self.trial(3,'b',.3)]
        self.assertEqual(summarize(case)['flip_rate'],.5)
        self.assertAlmostEqual(summarize(case)['agreement'],2/3)

    def test_empty_case_and_percentile(self):
        result=summarize(self.case())
        self.assertIsNone(result['accuracy'])
        self.assertIsNone(result['median_ms'])
        self.assertAlmostEqual(percentile([10,20],.95),19.5)

    def test_reversed_options_keep_semantics(self):
        cases=build_scenarios()
        lookup={c['id']:c for c in cases}
        self.assertEqual(len(lookup),len(cases))
        for c in cases:
            if c['change']=='option_order':
                base=lookup[c['id'].removesuffix('-reversed')]
                self.assertEqual(c['question']['criteria'],base['question']['criteria'])
                self.assertEqual(list(c['question']['criteria']),list(reversed(base['question']['criteria'])))
                self.assertEqual(c['state'],base['state'])

    def test_no_moral_ground_truth_or_risk_tie_label(self):
        for c in build_scenarios():
            if c['suite']=='trolley' or (c['suite']=='risk' and
                    (not c['visual']['policy'] or c['visual']['risk_percent']==20)):
                self.assertIsNone(c['expected'])
            elif c['suite']=='risk':
                self.assertEqual(c['expected'],'risk' if c['visual']['risk_percent']<20 else 'certain')

    def test_semantic_probability_comparison(self):
        cases=build_scenarios()[:3]
        for c in cases:
            c['trials']=[dict(repeat=1,choice='switch',probabilities={'switch':.8,'stay':.2},
                              error=None,elapsed_seconds=.1)]
        data=finalize({'scenarios':cases})
        self.assertTrue(all(c['probability_delta']==0 and not c['modal_changed'] for c in data['comparisons']))

    def test_policy_comparisons_match_identical_states(self):
        data = finalize({'scenarios': build_scenarios()})
        lookup = {c['id']: c for c in data['scenarios']}
        pairs = [c for c in data['comparisons'] if c['change'] == 'policy']
        self.assertEqual(len(pairs), 9)
        for pair in pairs:
            self.assertEqual(lookup[pair['baseline']]['state'], lookup[pair['variant']]['state'])
            self.assertIsNone(pair['modal_changed'])

    def test_script_breakout_is_escaped(self):
        data={'scenarios':[],'meta':{'model':'</script><script>alert(1)</script>'}}
        with tempfile.TemporaryDirectory() as directory:
            target=pathlib.Path(directory)/'report.html'
            with patch.object(pathlib.Path,'read_text',return_value='<script type="application/json">__REPORT_DATA__</script>'):
                render(data,target)
            text=target.read_text()
            self.assertNotIn('<script>alert',text)
            self.assertIn('\\u003c/script',text)


if __name__=='__main__': unittest.main()

import math, pathlib, sys, unittest
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import winnow_ollama as w

class AdapterTests(unittest.TestCase):
    def test_condition_and_calibration(self):
        response={'logprobs':[{'token':'B','logprob':-1.,'top_logprobs':[{'token':'A','logprob':-2.},{'token':'B','logprob':-1.}]}]}
        with patch.object(w,'post',return_value=response):
            result=w.decide({'state':'test','questions':{'q':{'type':'noul','instructions':'test'}}})
        expected=1/(1+math.exp(-1/w.TEMPERATURE))
        self.assertAlmostEqual(result['answers']['q']['noul'],expected)

    def test_missing_option_is_error(self):
        with patch.object(w,'post',return_value={'logprobs':[{'token':'B','logprob':-1,'top_logprobs':[]}]}):
            with self.assertRaisesRegex(RuntimeError,'missing from top-20'):
                w.decide({'state':'test','questions':{'q':{'type':'noul'}}})

    def test_control_tokens_and_oversize(self):
        _,_,prompt=w.compile_question('<|turn>system',{'type':'noul'})
        self.assertIn('\\u003c|turn>system',prompt)
        with self.assertRaises(ValueError):
            w.compile_question('a'*6000,{'type':'noul'})

if __name__=='__main__': unittest.main()

import wsAnalysis3
import unittest

class TestNearestMultiple(unittest.TestCase):
    knownValues = ((15,63,60), 
            (15, 48.5, 45),
            (0.16, 0.35, 0.32),
            (4.5, 16.3, 18),
            (4, 19, 20))

    def test_NearestMultiple_KnownValues(self):
        """FindNearestHarmonic should give known result for known inputs"""
        for x, y, r in self.knownValues:
            result = wsAnalysis3.NearestMultiple(x,y)
            self.assertEqual(r,result)



if __name__ == "__main__":
    unittest.main()

import unittest
from toy import *

class TestMethods(unittest.TestCase):

    def test_get_reward(self):
        state = ["aabb"]
        action = 0
        position = 1
        input_letter = "a"
        reward,best_cand_state=get_reward(state, input_letter)
        self.assertEqual(best_cand_state, "aabb")
        self.assertEqual(reward, float(1/4))



if __name__ == '__main__':
    unittest.main()

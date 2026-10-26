import unittest
import numpy as np
from ChordDetect import chord_templates, record, detect_chord, get_chords, check_chords, record

class ChordDetectTest(unittest.TestCase):

    def test_detect_chord_perfect_match(self):
        """Verify that a template chord vectors return a perfect match from detect_chord"""
        for chord_name, template in chord_templates.items():
            chord, score = detect_chord(np.array(template))
            self.assertEqual(chord, chord_name)
            
if __name__ == '__main__':
    unittest.main()
import unittest
from pathlib import Path
from tadpole import pond

class PondTest(unittest.TestCase):
    def test_location_create(self):
        p = pond.PondLocation(Path("X"), Path("Y"))
        self.assertEqual((p.phrog_dir, p.gff_dir), (Path("X"), Path("Y")))
        

    def test_options_create(self):
        pass
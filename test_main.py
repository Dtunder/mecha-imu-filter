import unittest
from unittest.mock import patch
import main

class TestMain(unittest.TestCase):
    @patch('builtins.print')
    def test_main(self, mock_print):
        with self.assertLogs('main', level='INFO') as cm:
            main.main()
            self.assertTrue(any("Starting mecha-imu-filter..." in log for log in cm.output))
        mock_print.assert_called_once_with("Hello from mecha-imu-filter!")

if __name__ == '__main__':
    unittest.main()

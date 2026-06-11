import unittest
from unittest.mock import patch
import main

class TestMain(unittest.TestCase):
    @patch('builtins.print')
    def test_main(self, mock_print):
        main.main()
        mock_print.assert_called_once_with("Hello from mecha-imu-filter!")

if __name__ == '__main__':
    unittest.main()

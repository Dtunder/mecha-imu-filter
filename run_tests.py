import unittest
import sys

def run_tests():
    """Runs all tests and outputs a summary report."""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=".", pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n=== TEST SUMMARY ===")
        print(f"Total tests run: {result.testsRun}")
        print("Status: ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n=== TEST SUMMARY ===")
        print(f"Total tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print("Status: SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()

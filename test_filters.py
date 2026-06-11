import unittest
import math
from filters import KalmanFilter, ComplementaryFilter, LowPassFilter

class TestFilters(unittest.TestCase):

    def test_validation(self):
        import logging
        # Test number validation (type, nan, inf)
        with self.assertLogs('filters', level='ERROR') as cm:
            with self.assertRaises(TypeError):
                KalmanFilter(initial_value="0.0")
            self.assertTrue(any("Validation failed" in log for log in cm.output))
            
            with self.assertRaises(ValueError):
                KalmanFilter(initial_value=math.nan)
            
            with self.assertRaises(ValueError):
                KalmanFilter(initial_value=math.inf)
                
            # Test specific ranges
            with self.assertRaises(ValueError):
                KalmanFilter(process_noise=-1.0)
            with self.assertRaises(ValueError):
                KalmanFilter(measurement_noise=0.0)
            with self.assertRaises(ValueError):
                KalmanFilter(measurement_noise=-1.0)
            with self.assertRaises(ValueError):
                KalmanFilter(estimated_error=-1.0)
                
            with self.assertRaises(ValueError):
                ComplementaryFilter(alpha=-0.1)
            with self.assertRaises(ValueError):
                ComplementaryFilter(alpha=1.1)
                
            cf = ComplementaryFilter()
            with self.assertRaises(ValueError):
                cf.update(10.0, 0.0, -1.0)
                
            with self.assertRaises(ValueError):
                LowPassFilter(alpha=-0.1)
            with self.assertRaises(ValueError):
                LowPassFilter(alpha=1.1)

    def test_kalman_zero_division(self):
        # We manually set parameters such that self.p + self.r = 0 (although r cannot be initialized to <= 0)
        kf = KalmanFilter()
        kf.p = -1.0
        kf.r = 1.0
        kf.q = 0.0
        # self.p + self.q = -1.0 + 0 = -1.0
        # self.p + self.r = -1.0 + 1.0 = 0.0
        
        with self.assertLogs('filters', level='WARNING') as cm:
            kf.update(10.0) # Shouldn't raise ZeroDivisionError
            self.assertTrue(any("ZeroDivisionError" in log for log in cm.output))

    def test_kalman_filter(self):
        kf = KalmanFilter(initial_value=0.0)
        
        # Test with a steady value
        val1 = kf.update(10.0)
        self.assertAlmostEqual(val1, 9.9, places=1) # The first update won't perfectly reach 10 due to noise config
        
        # After multiple updates, it should converge to 10
        for _ in range(100):
            val = kf.update(10.0)
            
        self.assertAlmostEqual(val, 10.0, places=2)
        
        # Test filter method is an alias
        kf1 = KalmanFilter(initial_value=0.0)
        kf2 = KalmanFilter(initial_value=0.0)
        self.assertEqual(kf1.update(10.0), kf2.filter(10.0))

    def test_complementary_filter(self):
        cf = ComplementaryFilter(alpha=0.9, initial_value=0.0)
        
        # Let's say we have an angle of 10 from accelerometer, and 0 from gyro over 1s
        accel_angle = 10.0
        gyro_rate = 0.0
        dt = 1.0
        
        val1 = cf.update(accel_angle, gyro_rate, dt)
        # angle = 0.9 * (0 + 0) + 0.1 * 10 = 1.0
        self.assertAlmostEqual(val1, 1.0, places=5)
        
        # Test filter method is an alias
        val2 = cf.filter(accel_angle, gyro_rate, dt)
        self.assertAlmostEqual(val2, 1.9, places=5) # 0.9 * (1.0 + 0) + 0.1 * 10 = 0.9 + 1 = 1.9

    def test_low_pass_filter(self):
        lpf = LowPassFilter(alpha=0.1, initial_value=0.0)
        
        # Update with value 10
        val1 = lpf.update(10.0)
        # value = 0.1 * 10 + 0.9 * 0 = 1.0
        self.assertAlmostEqual(val1, 1.0, places=5)
        
        # Test filter method is an alias
        val2 = lpf.filter(10.0)
        self.assertEqual(val2, lpf.value)
        # value = 0.1 * 10 + 0.9 * 1.0 = 1.0 + 0.9 = 1.9
        self.assertAlmostEqual(val2, 1.9, places=5)

if __name__ == '__main__':
    unittest.main()

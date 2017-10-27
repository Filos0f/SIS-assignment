# SIS-assignment

Variant C: Azimuth movement with gyroscope.

Description of the project:

In this work, the Lego ev3 set was used. The task is to use and work with the gyroscope sensor. Using this sensor, it is necessary to program the behavior of the working system to determine a certain angle for which it is necessary to deploy the system, after the turn the system must move in the given direction. After starting the system, the sensor senses the main axis as the main axis, with respect to which the deviation will be determined. If the system is rejected by a value greater than or less than the specified value, the system must return in the direction of its movement.

Explanation of the result:

As a result, a moving station was assembled from the standard elements of the Lego ev3 kit with a gyroscope sensor. The program code for controlling the motors of the running axle has been described. After the system is turned on, the gyro sensor records the axis along which it is oriented during the start. In the code, the readings from the sensor are taken and the test is performed until the angle of the gyroscope direction is deflected 90 degrees from the initial axis, one wheel is given a positive value, the other is negative, this allows the workstation to unfold in a certain direction. When the deviation has occurred, both motors are given the same values, which allows it to move in one direction.
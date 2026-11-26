# The TM171 IMU

The IMU on the back of the hand (SYD Dynamics TM171, on USB). Plain Python 3.8 or newer,
standard library only: nothing to install. Linux.

    real/imu/wlimu              live view
    real/imu/wlimu check        rate, losses, delay and noise, 20 s
    real/imu/wlimu axes         the seven moves that prove the axes, once it's on the hand
    real/imu/wlimu shake        how wrong its "down" gets when it's knocked around on a table
    real/imu/wlimu log          write the readings to outputs/real/imu/logs/*.csv
    real/imu/wlimu settings     its settings against the ones we want (changes nothing)
    real/imu/wlimu apply        give it the settings we want, until it's unplugged
    real/imu/wlimu apply --save the same, in its permanent memory
    real/imu/wlimu set odr=200  change one (names: tm171.py, _SWITCHES and _NUMBERS)
    real/imu/wlimu restore FILE put back a backup (one is written to outputs/real/imu/settings/ before every change)

One program at a time on the IMU: close the live view before the loop or anything else
reads it. It opens no port but the IMU's (an STM32 virtual COM port in
`/dev/serial/by-id/`; `TM171_SERIAL` picks one if several are plugged in). The user needs
access to that port (the `dialout` group, or a udev rule).

In a control loop: `from tm171 import TM171`, see the top of tm171.py. It gives the gyro
(rad/s) and gravity (unit vector) in the palm's axes, the same as in the simulation:
the IMU sits turned by 90 degrees on its mount, and tm171.py turns its readings.

After plugging in, leave it still: its quality grade goes to 4 in about half a minute and
to 5 in a few minutes. Don't run a policy under 4.

Checks that need no IMU: `uv run pytest tests/test_tm171.py`.

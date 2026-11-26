# Training the next policy

What the first policies in simulation (README, "First runs") and the scripted runs on
the real hand ([real_hand.md](real_hand.md)) say the next training batch should change.
Each number says where it comes from: **measured** on the real hand, **sim** (simulation
only), or **guess**.

## In short

1. **Train on the hand as it is.** Stock fingertips (no socks for now), LEAP's own thumb
   limits, the IMU and its holder, the 350 mA current limit. `as_built=True` still adds
   the TPU socks: it needs a way to leave them off.
2. **Imitate the scripted crawl instead of rewarding walking from scratch.** Four reward
   sets each found a way to keep a finger in the air. The crawl already moves the real
   hand forward: played open loop, six cycles took it past a 12 cm object (measured
   once; it ran twice without a stop), and three cycles finished on 45 of 45 random simulated hands.
3. **Turn on torsional friction at the contacts.** With `condim=3` a fingertip can spin
   on the floor for free, and the simulated hand turns the wrong way while crawling.
   With 0.003 m at the fingertips and 0.03 m on the other contacts, the heading error
   per clip halves (10° to 5° on average over five real runs) and the crawl's heading
   matches within about 1°. The tilt match doesn't change.
4. **Center the motors and friction on what the hand showed,** and keep randomizing:
   the real hand differs from run to run more than the simulator does (standing up, it
   turned anywhere from −8° to −22° for the same commands).
5. **Measure the motors' torque per amp** before trusting the 0.181 N·m cap.

## The model: this repository against the real hand

| Setting | In the repository | The real hand | For the next batch |
| --- | --- | --- | --- |
| Fingertips | `as_built=True` adds the TPU socks; the randomized friction acts on them (`FOOT_GEOMS`) | stock fingertips, no socks | an as-built hand without socks, its friction randomized on the tips |
| Thumb limits | LEAP's own (`THUMB_LIMITS`) | each thumb joint reaches its limit at about 100 mA (measured) | keep |
| IMU and holder | 58 g on the back, the IMU turned as mounted | the simulated "down" stays within about 1.7° of the real IMU's over every run (measured): its axes and turn are right | keep; weigh it |
| Motor stiffness | 3.0 N·m/rad; `-Motion-Flat-v1` scales it by 0.6 to 1.2 | the joints follow their commands like a 2.2-3.0 N·m/rad servo at gains 1200 / 0 / 600, 2.6 fitting best; 1.6 is ruled out (measured) | scale 0.7 to 1.1 (2.1-3.3), or keep the wider range |
| Torque cap | 0.181 N·m: 350 mA through the datasheet's stall ratio; scaled 0.8 to 1.2 | not measured. The stiffness fit suggests more torque per mA than that ratio | keep 0.8 to 1.2 until measured |
| Sliding friction | tips 0.5 (Menagerie), socks 0.6; socks randomized 0.4 to 0.9 | stock tips on a wooden table: below 0.5 the simulated stand-up goes wrong where the real one doesn't; above it, slow moves can't tell (measured against sim) | 0.5 to 1.0 on the tips |
| Torsional friction | none (`condim=3` on the hand's contacts) | crawling, the real hand turns −5° to −7°; the simulation +5° to +10° (measured against sim) | `condim=4`; tips 0.001 to 0.005 m, other contacts 0.01 to 0.03 m, randomized (below) |
| Delay | up to 10 physics steps (20 ms) in `-v1` | motors read at 100 Hz, targets sent one tick (10 ms) later; the loop is typically 0.2 ms late, at worst 7.5; about 0.35 % of motor reads are lost (measured) | keep |
| IMU noise | gyro ±0.05 rad/s, gravity ±0.03 in `-v1` | `hello` held with 50 times the real IMU's measured noise (sim) | keep |
| How the weight is shared, standing | spread more evenly over the four digits (sim) | by the motor currents, the index carries almost nothing, the middle and ring fingers most (measured) | keep the palm's center of mass randomized (±8 mm now); measure the hand's balance point |
| Roll when one finger lifts | about twice the real roll (sim) | measured with the IMU | not fixed yet: probably the weight sharing and the stiffness |
| Cables | random pulls up to 0.15 N and 0.01 N·m on the holder and the palm (guess) | not measured | keep |
| Mass | 0.746 kg (no official figure) plus the parts | not weighed | weigh the hand with everything on |

## Torsional friction: the test

Each real run's exact motor commands were replayed in plain MuJoCo from the real starting
pose (stiffness 2.6 N·m/rad, sliding friction 0.8), with the contacts at `condim=4` and a
torsional coefficient, against `condim=3`. Real heading: the IMU's gyro integrated over
each clip. Simulated: the palm's yaw. Degrees.

| Clip (real runs) | Real | Sim, no torsion | Sim, tips 0.003 m | Sim, tips 0.003 m, other contacts 0.03 m |
| --- | --- | --- | --- | --- |
| stand up (5) | −21.6, −12.6, −12.2, −17.6, −8.1 | −22.6 to −23.0 | −22.0 to −22.5 | −18.1 to −18.5 |
| lie down (5) | +19.1, +16.0, +17.3, +17.9, +11.2 | +31.2 to +31.8 | +31.1 to +31.5 | +24.0 to +24.3 |
| crawl, 3 cycles | −4.8 | +4.7 | −3.7 | −3.8 |
| crawl, 6 cycles (2) | −6.6, −5.2 | +9.6, +9.6 | −6.7, −6.4 | −6.7, −6.4 |
| push-ups | 0.0 | −0.5 | −4.7 | −5.9 |
| sway | −0.8 | −0.1 | −6.2 | −6.2 |
| taps | +0.9 | +0.1 | −1.6 | −1.6 |
| **error per clip, on average** | | **10.0** | **8.1** | **5.1** |
| "down" off, on average | | 1.68 | 1.70 | 1.70 |

- **What it shows:** most of the heading gap is friction the model leaves out. The crawl
  turns the right way and by the right amount as soon as the fingertips resist spinning.
  Standing up and lying down drag the palm and the knuckles on the table, which is why
  the other contacts matter there.
- **What it doesn't:** one coefficient doesn't fit everything. Push-ups and sway, which
  don't turn the real hand, turn the simulated one 5-6° with torsion on, and lying down
  still turns about 8° too far. More torsion at the tips is worse: at 0.01 m the crawl turns
  −16° to −31°. Hence a range, not a value.
- The sliding friction and stiffness above were the best fit over all runs; the "down"
  error doesn't depend on them much (see real_hand.md).

## The task: imitate the crawl

- **Why not rewards alone:** README "First runs", four reward sets on the as-built hand,
  four ways of not walking (three fingers and a thumb in the air, balancing on two
  digits, the index held up for good, a slow crawl with the index up 95 % of the time).
- **The crawl** (real_hand.md, "How the scripts were made") is quasi-static: the palm
  moves forward over the fingers, the thumb is dragged forward, then each finger in turn
  lifts 9 mm and steps 23 mm. About 9 s a cycle, written with the planted-feet
  kinematics `clips.py` uses for `hello`, on stock fingertips. Open loop: 14.7 cm for 6
  cycles in simulation, past 12 cm on the real hand; 45 of 45 random simulated hands
  finished 3 cycles (63 to 95 mm).
- **What's needed:** the crawl as a motion file for `Walking-Leap-Motion-Flat-*` (the
  tool that wrote it is outside this repository for now: it writes joint targets at 50
  Hz and the IMU reading the simulation expects), the hand model above, then the `-v1`
  randomization with the changes above. Faster or steered walking can come after,
  starting from the imitation policy.
- **`hello` was written for the socks.** On stock fingertips, played open loop, 24 of
  45 random simulated hands finish it; its first 5 s (two push-ups and a stretch)
  finish 45 of 45. Either the socks go on before `hello` drives the real hand, or the
  clip is written again for stock tips.

## Before a policy drives the real hand

1. The randomized test table of README "First runs", on the new model, with rows for
   torsion and the stiffness range above.
2. The real logs replayed through the policy (it computes on the recorded sensors).
3. On the hand, in the shadow: the policy computes from the live sensors and sends
   nothing (done for `hello`: 0.046 rad from the clip, 0.048 in simulation).
4. Then live, from the stand, with somebody at the power switch.

## What to measure next, most useful first

1. **Torque per amp** of the XC330 at 350 mA: a known weight on a known lever. The
   torque cap and the stiffness both rest on the datasheet's ratio.
2. **The hand's weight and balance point** with the IMU and its holder on.
3. **Fingertip friction on the table:** the tilt test for sliding; for turning, a
   fingertip pressed down and twisted.
4. **The cables' pull** at the IMU's plug and at the wrist.

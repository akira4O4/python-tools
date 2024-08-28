import time


def pid():
    x_target = 100
    x_curr = 40
    x_static_difference = 6

    kp = 4
    ki = 0.5
    kd = 1.0

    dt = 0.1
    integral = 0
    previous_error = 0  # 初始化前一次误差为0

    times = 1000
    for t in range(times):
        error = x_target - x_curr
        integral += error * dt
        derivative = (error - previous_error) / dt

        # p
        p_output = kp * error
        # I
        i_output = ki * integral
        # D
        d_output = kd * derivative

        # x_curr = x_curr + p_output * dt - x_static_difference * dt
        # x_curr = x_curr + (p_output + i_output) * dt - x_static_difference * dt
        x_curr = x_curr + (p_output + i_output + d_output) * dt - x_static_difference * dt

        previous_error = error

        print(f'Error: {error}')
        print(f'Integral: {integral}')
        print(f'Derivative: {derivative}')
        print(f'x_curr: {x_curr}\n')


if __name__ == '__main__':
    pid()

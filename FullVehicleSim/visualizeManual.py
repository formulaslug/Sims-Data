# Vibe coded lol
import pygame
import numpy as np
import asyncio
import platform
from state import VehicleState
from engine import stepState

#Simulated control inputs (since no file I/O in Pyodide)
timeBasedInputs = sorted([
   (2.0, [0.75, 0, 0.4]),
   (20.3, [0.0, 1.0, 0.0])
])

def setup():
    global screen, clock, car_surface, font, currVehicle, simulation_time, currInput, timeSinceInput, initSpeed, input_index, past_positions, start_time, scale, car_length, car_width, grid_spacing, FPS, stepsPerSecond, delta, simDuration, screen_width, screen_height

    # Initialize Pygame
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Vehicle Dynamics Visualization")
    clock = pygame.time.Clock()

    # Simulation parameters
    stepsPerSecond = 100
    delta = 1 / stepsPerSecond
    simDuration = 25
    FPS = 60

    # Initialize vehicle state
    currVehicle = VehicleState(
        stepSize = 1/stepsPerSecond,
        position=np.asarray([0,0,0], dtype=np.float32),
        speed=0,
        acceleration=0,
        heading = np.asarray([1,0,0], dtype=np.float32),
        charge=50,
        lastCurrent=0,
        throttle = 0,
        brakes = 0,
        yawRate = 0,
        steerAngle = 0,
        brakeTemperature = 150,
        timeSinceLastSteer = 0,
        initSpeed = 0,
    )

    # Visualization parameters
    scale = 10  # pixels per meter
    car_length = 4 * scale
    car_width = 2 * scale
    grid_spacing = 10.0  # meters

    # Create car surface
    car_surface = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
    pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, car_length, car_width))

    # Font for telemetry
    font = pygame.font.SysFont('Arial', 16)

    # Main loop variables
    simulation_time = 0.0
    currInput = [0, 0, 0]
    timeSinceInput = 0.0
    initSpeed = 0.0
    input_index = 0
    past_positions = []
    start_time = pygame.time.get_ticks() / 1000.0

def update_loop():
    global currVehicle, simulation_time, currInput, timeSinceInput, initSpeed, input_index, past_positions, start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return

    current_time = pygame.time.get_ticks() / 1000.0
    elapsed = current_time - start_time
    while simulation_time < elapsed and simulation_time < simDuration:
        while input_index < len(timeBasedInputs) and simulation_time >= timeBasedInputs[input_index][0]:
            currInput = timeBasedInputs[input_index][1]
            timeSinceInput = 0.0
            initSpeed = max(currVehicle.speed, 5)
            input_index += 1

        print(currVehicle.yawRate, initSpeed, timeSinceInput)
        currVehicle = stepState(currVehicle, currInput, delta, timeSinceInput, initSpeed)
        simulation_time += delta
        timeSinceInput += delta

        if int(simulation_time * 10) > int((simulation_time - delta) * 10):
            past_positions.append(currVehicle.position.copy())

    # Clear screen
    screen.fill((255, 255, 255))

    # Draw grid
    car_x, car_y = currVehicle.position[0], currVehicle.position[1]
    x_min = car_x - (screen_width / 2) / scale
    x_max = car_x + (screen_width / 2) / scale
    y_min = car_y - (screen_height / 2) / scale
    y_max = car_y + (screen_height / 2) / scale

    k_min = int(np.ceil(x_min / grid_spacing))
    k_max = int(np.floor(x_max / grid_spacing))
    m_min = int(np.ceil(y_min / grid_spacing))
    m_max = int(np.floor(y_max / grid_spacing))

    for k in range(k_min, k_max + 1):
        world_x = k * grid_spacing
        screen_x = screen_width / 2 + (world_x - car_x) * scale
        pygame.draw.line(screen, (200, 200, 200), (screen_x, 0), (screen_x, screen_height), 1)

    for m in range(m_min, m_max + 1):
        world_y = m * grid_spacing
        screen_y = screen_height / 2 - (world_y - car_y) * scale
        pygame.draw.line(screen, (200, 200, 200), (0, screen_y), (screen_width, screen_y), 1)

    # Draw trail
    for pos in past_positions:
        screen_pos = (
            screen_width / 2 + (pos[0] - car_x) * scale,
            screen_height / 2 - (pos[1] - car_y) * scale
        )
        pygame.draw.circle(screen, (0, 0, 255), screen_pos, 2)

    # Draw car
    angle = np.degrees(np.arctan2(currVehicle.heading[1], currVehicle.heading[0]))
    rotated_car = pygame.transform.rotate(car_surface, angle)
    rect = rotated_car.get_rect()
    rect.center = (screen_width / 2, screen_height / 2)
    screen.blit(rotated_car, rect)

    # Draw telemetry
    telemetry_texts = [
        f"Time: {simulation_time:.2f} s",
        f"Speed: {currVehicle.speed:.2f} m/s",
        f"Position: ({currVehicle.position[0]:.2f}, {currVehicle.position[1]:.2f}) m",
        f"Heading: ({currVehicle.heading[0]:.2f}, {currVehicle.heading[1]:.2f}) m",
        f"Throttle: {currVehicle.throttle:.2f}",
        f"Brakes: {currVehicle.brakes:.2f}",
        f"Steer Angle: {np.degrees(currVehicle.steerAngle):.2f}°",
        f"Yaw Rate: {np.degrees(currVehicle.yawRate):.2f} deg/s"
    ]
    for i, text in enumerate(telemetry_texts):
        text_surface = font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10 + i * 20))

    pygame.display.flip()

async def main():
    setup()
    while simulation_time < simDuration:
        update_loop()
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())

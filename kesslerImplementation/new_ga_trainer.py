import EasyGA
import sys
import os 
from kesslergame import Scenario, GraphicsType, TrainerEnvironment
from IrisController import IrisController, DefaultController
import random
import math 
from kesslergame import KesslerController

def test_scenario():
    my_test_scenario = Scenario(
        name='Test Scenario',
        num_asteroids=30,
        #num_mines=3,
        ship_states=[
            {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1},
           # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2},
        ],
        map_size=(1000, 800),
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )
    return my_test_scenario

def game_env():
    game_settings = {
        'perf_tracker': True,
        'graphics_type': GraphicsType.Tkinter,
        'realtime_multiplier': 1,
        'graphics_obj': None
    }
    game = TrainerEnvironment(settings=game_settings)
    return game

def generate_chromosome():
    # Ensure ordered and non-overlapping gene values
    # Theta delta
    theta_delta_nl = random.uniform(-math.pi, -2 * math.pi / 90)  # Adjusted upper bound
    theta_delta_nm = random.uniform(theta_delta_nl + 0.01, -math.pi / 90)
    theta_delta_ns = random.uniform(theta_delta_nm + 0.01, math.pi / 90)
    theta_delta_z = random.uniform(theta_delta_ns + 0.01, math.pi / 90) 
    theta_delta_ps = random.uniform(theta_delta_z + 0.01, 2 * math.pi / 90)
    theta_delta_pm = random.uniform(theta_delta_ps + 0.01, math.pi / 30)
    theta_delta_pl = random.uniform(theta_delta_pm + 0.01, math.pi / 30 + 0.1) 

    # Bullet times
    bullet_time_s = random.uniform(0, 0.05)
    bullet_time_m = random.uniform(bullet_time_s + 0.001, 0.1)
    bullet_time_l = random.uniform(bullet_time_m + 0.001, 1)

    ship_turn_nl = random.uniform(-180, -150)
    ship_turn_nm = random.uniform(ship_turn_nl + 1, -90)
    ship_turn_ns = random.uniform(ship_turn_nm + 1, -10)  # Adjusted upper bound to leave space for ship_turn_z

    ship_turn_z = random.uniform(-10, 10)  # Neutral ship turn around 0 degrees

    ship_turn_ps = random.uniform(ship_turn_z + 1, 90)  # Start after ship_turn_z
    ship_turn_pm = random.uniform(ship_turn_ps + 1, 150)
    ship_turn_pl = random.uniform(ship_turn_pm + 1, 180)

    # Distances
    distance_1 = random.uniform(0, 250)
    distance_2 = random.uniform(distance_1 + 1, 450)
    distance_3 = random.uniform(distance_2 + 1, 650)
    distance_4 = random.uniform(distance_3 + 1, 850)
    distance_5 = random.uniform(distance_4 + 1, 1000)

    # Relative directions
    relative_dir_f = random.uniform(0, 30)
    relative_dir_df = random.uniform(relative_dir_f + 1, 90)
    relative_dir_s = random.uniform(relative_dir_df + 1, 150)
    relative_dir_db = random.uniform(relative_dir_s + 1, 180)
    relative_dir_b = random.uniform(relative_dir_db + 1, 360)

    # Ship speeds
    speed_vsn = random.uniform(-240, -180)
    speed_sn = random.uniform(speed_vsn + 1, -90)
    speed_z = random.uniform(speed_sn + 1, 0)
    speed_sp = random.uniform(0, 90)
    speed_fp = random.uniform(speed_sp + 1, 240)

    # Thrust
    thrust_close = random.uniform(-150, -50)
    thrust_medium = random.uniform(thrust_close + 1, 50)
    thrust_far = random.uniform(thrust_medium + 1, 150)
    #mines 
    
    return [
        thrust_close, thrust_medium, thrust_far,
        relative_dir_f, relative_dir_df, relative_dir_s, relative_dir_db, relative_dir_b,
        distance_1, distance_2, distance_3, distance_4, distance_5,
        ship_turn_nl, ship_turn_nm, ship_turn_ns, ship_turn_z,ship_turn_ps, ship_turn_pm, ship_turn_pl,
        bullet_time_s, bullet_time_m, bullet_time_l,
        theta_delta_nl,theta_delta_nm,theta_delta_ns, theta_delta_z, theta_delta_ps,theta_delta_pm, theta_delta_pl,
        speed_vsn, speed_sn, speed_z, speed_sp, speed_fp
    ]

def fitness(chromosome):
    print("Evaluating Chromosome:", chromosome.gene_value_list)
    game = game_env()
    test_game = test_scenario()
    gene_values = chromosome.gene_value_list[0]

    # Initialize controller with the chromosome
    my_controller = IrisController(gene_values)

    # Run the game with the controller
    score, perf_data = game.run(scenario=test_game, controllers=[my_controller])

    # Extract the number of asteroids hit by team 1
    asteroids_hit = next((team.asteroids_hit for team in score.teams if team.team_id == 1), 0)
    print("Asteroids Hit:", asteroids_hit)

    return asteroids_hit

def EasyGa():
    ga = EasyGA.GA()
    ga.gene_impl = lambda: generate_chromosome()
    ga.chromosome_length = 1
    ga.population_size = 20
    ga.target_fitness_type = 'max'
    ga.generation_goal = 50
    ga.fitness_function_impl = fitness
    ga.evolve()
    ga.print_best_chromosome()

# Run the genetic algorithm
EasyGa()

import sys
import os 
from kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz 
import math 
import numpy as np 
from skfuzzy import control as ctrl 

class IrisController(KesslerController):
    def __init__(self):
        super().__init__()
        self.eval_frames = 0  # Frame counter
        best_genes = [-128.3405097788583, -91.19321590182416, 78.80766125152594, 
                       3.8547809989496473, 17.105524095419874, 54.024265488248666, 174.82365234755173,
                         336.4555025195891, 108.11337292236512, 410.1773585534388, 577.2029261809271, 
                         644.3264916787837, 960.7333653391574, -169.62617868106923, -117.15970361662664,
                        -80.46129062858716, 8.792284418592331, 59.39686621526819, 64.25482789301321,
                         112.63481671136218, 0.003728496354582228, 0.055466491962131215, 0.9210391974252481, 
                        -0.11840033086322954, -0.10613084204824573, -0.06902160017372097, 
                        -0.03961652615389806, 0.04942361071177715, 0.08702213407022384,
                          0.12829859903482038, -213.35375803886618,
                        -134.1609260168026, -56.01365854179761, 33.57427031492191, 123.33729140972838]
        # Unpack chromosome genes (34 genes now)
        (
            thrust_close, thrust_medium, thrust_far,
            relative_dir_f, relative_dir_df, relative_dir_s, relative_dir_db, relative_dir_b,
            distance_1, distance_2, distance_3, distance_4, distance_5,
            ship_turn_nl, ship_turn_nm, ship_turn_ns,ship_turn_z, ship_turn_ps, ship_turn_pm, ship_turn_pl,
            bullet_time_s, bullet_time_m, bullet_time_l,
            theta_delta_nl, theta_delta_nm, theta_delta_ns, theta_delta_z, theta_delta_ps, theta_delta_pm, theta_delta_pl,
            speed_vsn, speed_sn, speed_z, speed_sp, speed_fp
        ) = best_genes

        # Define fuzzy variables
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.01), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')

        ship_speed = ctrl.Antecedent(np.linspace(-240, 240, 1001), 'ship_speed')
        ast_dist = ctrl.Antecedent(np.arange(0, 1001, 1), 'ast_dist')
        
        # Thurst antecedent
        thrust = ctrl.Consequent(np.linspace(-500, 500, 10001), 'thrust')

        # Define membership functions using genes

        # Bullet time
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, bullet_time_s])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, bullet_time_s, bullet_time_m])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, bullet_time_m, bullet_time_l)
        
        # Theta delta
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, theta_delta_nl, theta_delta_nm)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [theta_delta_nl, theta_delta_nm, theta_delta_ns])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [theta_delta_nm, theta_delta_ns, theta_delta_z])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [theta_delta_ns, theta_delta_z, theta_delta_ps])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [theta_delta_z, theta_delta_ps, theta_delta_pm])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [theta_delta_ps, theta_delta_pm, theta_delta_pl])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, theta_delta_pm, theta_delta_pl)

        # Ship turn
        ship_turn['NL'] = fuzz.zmf(ship_turn.universe, ship_turn_nl, ship_turn_nm)
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [ship_turn_nl, ship_turn_nm, ship_turn_ns])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [ship_turn_nm, ship_turn_ns, ship_turn_ps])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [ship_turn_ns, ship_turn_z, ship_turn_ps])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [ship_turn_ns, ship_turn_ps, ship_turn_pm])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [ship_turn_ps, ship_turn_pm, ship_turn_pl])
        ship_turn['PL'] = fuzz.smf(ship_turn.universe, ship_turn_pm, ship_turn_pl)

        # Ship fire
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1]) 

        # Ship speed
        ship_speed['FR'] = fuzz.zmf(ship_speed.universe, -240, -150)
        ship_speed['R'] = fuzz.gaussmf(ship_speed.universe, -125, 50)
        ship_speed['S'] = fuzz.gaussmf(ship_speed.universe, 0, 60)
        ship_speed['F'] = fuzz.gaussmf(ship_speed.universe, 125, 50)
        ship_speed['FF'] = fuzz.gaussmf(ship_speed.universe, 150, 240)

        # Asteroid distance
        ast_dist['VC'] = fuzz.zmf(ast_dist.universe, 0, 150)
        ast_dist['C'] = fuzz.gaussmf(ast_dist.universe, 225, 100)
        ast_dist['M'] = fuzz.gaussmf(ast_dist.universe, 500, 150)
        ast_dist['F'] = fuzz.gaussmf(ast_dist.universe, 775, 100)
        ast_dist['VF'] = fuzz.smf(ast_dist.universe, 850, 1000)

        # Thrust
        thrust['FR'] = fuzz.zmf(thrust.universe, -500, -250)
        thrust['R'] = fuzz.gaussmf(thrust.universe, -250, 100)
        thrust['S'] = fuzz.gaussmf(thrust.universe, 0, 125)
        thrust['F'] = fuzz.gaussmf(thrust.universe, 250, 100)
        thrust['FF'] = fuzz.smf(thrust.universe, 250, 500)

        # Mine handling variables
        mine_danger = ctrl.Antecedent(np.arange(0, 2, 1), 'mine_danger')
        mine_danger['Safe'] = fuzz.trimf(mine_danger.universe, [0, 0, 1])
        mine_danger['Danger'] = fuzz.trimf(mine_danger.universe, [0, 1, 1])

        deploy_mine = ctrl.Consequent(np.arange(-1, 2, 1), 'deploy_mine')
        deploy_mine['Hold'] = fuzz.trimf(deploy_mine.universe, [-1, -1, 0])
        deploy_mine['Deploy'] = fuzz.trimf(deploy_mine.universe, [0, 1, 1])

        # Define rules for the targeting controller
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule5 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule6 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule7 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule10 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule11 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

       
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
        
        # Mine rules
        rule_mine1 = ctrl.Rule(ast_dist['VC'] & mine_danger['Danger'], deploy_mine['Deploy'])
        rule_mine2 = ctrl.Rule(ast_dist['M'] & mine_danger['Safe'], deploy_mine['Hold'])
        rule_mine3 = ctrl.Rule(ast_dist['F'] & mine_danger['Safe'], deploy_mine['Hold'])

        # Add rules to targeting_control
        self.targeting_control = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
            rule9, rule10, rule11, rule12, rule_mine1, rule_mine2, rule_mine3,
            rule13, rule14, rule15, rule16, rule17, rule18,
            rule19, rule20, rule21
        ])

        # Define thrust rules (comprehensive set)
        thrust_rule1 = ctrl.Rule(ast_dist['VC'] & ship_speed['FR'], thrust['FF'])
        thrust_rule2 = ctrl.Rule(ast_dist['VC'] & ship_speed['R'], thrust['FF'])
        thrust_rule3 = ctrl.Rule(ast_dist['VC'] & ship_speed['S'], thrust['S'])
        thrust_rule4 = ctrl.Rule(ast_dist['VC'] & ship_speed['F'], thrust['FR'])
        thrust_rule5 = ctrl.Rule(ast_dist['VC'] & ship_speed['FF'], thrust['FR'])
        
        thrust_rule6 = ctrl.Rule(ast_dist['C'] & ship_speed['FR'], thrust['FF'])
        thrust_rule7 = ctrl.Rule(ast_dist['C'] & ship_speed['R'], thrust['FF'])
        thrust_rule8 = ctrl.Rule(ast_dist['C'] & ship_speed['S'], thrust['F'])
        thrust_rule9 = ctrl.Rule(ast_dist['C'] & ship_speed['F'], thrust['R'])
        thrust_rule10 = ctrl.Rule(ast_dist['C'] & ship_speed['FF'], thrust['R'])
        
        thrust_rule11 = ctrl.Rule(ast_dist['M'] & ship_speed['FR'], thrust['FF'])
        thrust_rule12 = ctrl.Rule(ast_dist['M'] & ship_speed['R'], thrust['FF'])
        thrust_rule13 = ctrl.Rule(ast_dist['M'] & ship_speed['S'], thrust['FF'])
        thrust_rule14 = ctrl.Rule(ast_dist['M'] & ship_speed['F'], thrust['FF'])
        thrust_rule15 = ctrl.Rule(ast_dist['M'] & ship_speed['FF'], thrust['FF'])
        
        thrust_rule16 = ctrl.Rule(ast_dist['F'] & ship_speed['FR'], thrust['FF'])
        thrust_rule17 = ctrl.Rule(ast_dist['F'] & ship_speed['R'], thrust['FF'])
        thrust_rule18 = ctrl.Rule(ast_dist['F'] & ship_speed['S'], thrust['FF'])
        thrust_rule19 = ctrl.Rule(ast_dist['F'] & ship_speed['F'], thrust['FF'])
        thrust_rule20 = ctrl.Rule(ast_dist['F'] & ship_speed['FF'], thrust['FF'])
        
        thrust_rule21 = ctrl.Rule(ast_dist['VF'] & ship_speed['FR'], thrust['FF'])
        thrust_rule22 = ctrl.Rule(ast_dist['VF'] & ship_speed['R'], thrust['FF'])
        thrust_rule23 = ctrl.Rule(ast_dist['VF'] & ship_speed['S'], thrust['FF'])
        thrust_rule24 = ctrl.Rule(ast_dist['VF'] & ship_speed['F'], thrust['FF'])
        thrust_rule25 = ctrl.Rule(ast_dist['VF'] & ship_speed['FF'], thrust['FF'])

        # Add all thrust rules to thrust_control
        self.thrust_control = ctrl.ControlSystem([
            thrust_rule1, thrust_rule2, thrust_rule3, thrust_rule4, thrust_rule5,
            thrust_rule6, thrust_rule7, thrust_rule8, thrust_rule9, thrust_rule10,
            thrust_rule11, thrust_rule12, thrust_rule13, thrust_rule14, thrust_rule15,
            thrust_rule16, thrust_rule17, thrust_rule18, thrust_rule19, thrust_rule20,
            thrust_rule21, thrust_rule22, thrust_rule23, thrust_rule24, thrust_rule25           
        ])
        
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # Find the closest asteroid
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None
        drop_mine = False  # Default value

        for a in game_state["asteroids"]:
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None or closest_asteroid["dist"] > curr_dist:
                closest_asteroid = dict(aster=a, dist=curr_dist)

        if closest_asteroid is None:
            # No asteroids present
            return 0.0, 0.0, False, False

        # Calculate intercept parameters
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)
        ast_head = (180 / math.pi) * math.atan2(-asteroid_ship_x, -asteroid_ship_y)

        # Adjust ship heading
        adjusted_heading = ship_state["heading"] % 360
        if adjusted_heading > 180:
            adjusted_heading -= 360

        diff_heading = abs(ast_head - adjusted_heading) % 360

        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1],
                                        closest_asteroid["aster"]["velocity"][0])
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        asteroid_vel = math.hypot(closest_asteroid["aster"]["velocity"][0],
                                  closest_asteroid["aster"]["velocity"][1])
        bullet_speed = 800  # Hard-coded bullet speed

        # Determinant of the quadratic formula b^2 - 4ac
        a_quadratic = asteroid_vel**2 - bullet_speed**2
        b_quadratic = -2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2
        c_quadratic = closest_asteroid["dist"]**2
        discriminant = b_quadratic**2 - 4 * a_quadratic * c_quadratic

        if discriminant >= 0 and a_quadratic != 0:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b_quadratic + sqrt_discriminant) / (2 * a_quadratic)
            t2 = (-b_quadratic - sqrt_discriminant) / (2 * a_quadratic)
            bullet_t = min(filter(lambda t: t >= 0, [t1, t2]), default=None)
        else:
            bullet_t = None

        if bullet_t is None:
            bullet_t = 0.0  # Default value if intercept time can't be calculated

        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y), (intrcpt_x - ship_pos_x))

        # Mine danger assessment
        mine_danger_value = 0  # Default safe value
        for mine in game_state.get("mines", []):
            distance_to_mine = math.hypot(
                ship_pos_x - mine["position"][0],
                ship_pos_y - mine["position"][1]
            )
            if distance_to_mine < 150:  # Threshold for mine danger
                mine_danger_value = 1  # Mark as dangerous
                break

        shooting_theta = my_theta1 - math.radians(ship_state["heading"])
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Targeting controller
        shooting = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.input['mine_danger'] = mine_danger_value
        shooting.input['ast_dist'] = closest_asteroid["dist"]  # Ensure this input is set

        try:
            shooting.compute()
            turn_rate = shooting.output['ship_turn']
            fire = shooting.output['ship_fire'] >= 0
            drop_mine = shooting.output.get('deploy_mine', 0) > 0
        except Exception as e:
            print(f"Exception during targeting computation: {e}")
            turn_rate = 0.0
            fire = False
            drop_mine = False

        # Movement controller
        movement = ctrl.ControlSystemSimulation(self.thrust_control, flush_after_run=1)
        movement.input['ast_dist'] = closest_asteroid["dist"]
        movement.input['ship_speed'] = ship_state['speed']

        try:
            movement.compute()
            thrust = movement.output['thrust']
            
            if thrust > 480:
                thrust = 480
            if thrust < -480:
                thrust = -480
        except Exception as e:
            print(f"Exception during movement computation: {e}")
            thrust = 0.0

        self.eval_frames += 1

        #DEBUG
        print(f"Thrust: {thrust}, Bullet Time: {bullet_t}, Shooting Theta: {shooting_theta}, Turn Rate: {turn_rate}, Fire: {fire}, Drop Mine: {drop_mine}")

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "IrisController"
    
class DefaultController(KesslerController):
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        return 0.0, 0.0, False

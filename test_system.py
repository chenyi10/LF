"""Quick test script to verify the system works"""

import sys
import numpy as np

print("Testing Leader-Follower Navigation System...")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from lf_mha_multi.envs.nav_env import LeaderFollowerNavEnv
    from lf_mha_multi.envs.obstacles import ObstacleMap, Wall
    from lf_mha_multi.envs.sensors import RaySensor
    from lf_mha_multi.controllers.obstacle_avoid import ObstacleAvoidanceController
    from lf_mha_multi.controllers.formation_pd import FormationPDController
    from lf_mha_multi.utils.config import load_config
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load configuration
print("\n2. Testing configuration...")
try:
    config = load_config()
    print(f"✓ Config loaded: {config['env']['n_agents']} agents, {config['leader']['n_hypotheses']} hypotheses")
except Exception as e:
    print(f"✗ Config load failed: {e}")
    sys.exit(1)

# Test 3: Create obstacle map
print("\n3. Testing obstacle map...")
try:
    obs_map = ObstacleMap(config['obstacles'])
    print(f"✓ Obstacle map created with {len(obs_map.walls)} walls")
    
    # Test collision detection
    collision = obs_map.check_collision_circle(np.array([5.0, 0.5]), 0.3)
    print(f"✓ Collision detection works (collision={collision})")
    
    # Test ray casting
    dist, hit_point, normal = obs_map.cast_ray(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), 10.0
    )
    print(f"✓ Ray casting works (distance={dist:.2f})")
except Exception as e:
    print(f"✗ Obstacle map test failed: {e}")
    sys.exit(1)

# Test 4: Test sensors
print("\n4. Testing sensors...")
try:
    sensor = RaySensor(n_rays=8, max_range=3.0)
    distances, directions, normals = sensor.sense(
        np.array([0.0, 0.0]), obs_map, config['env']['world_size']
    )
    print(f"✓ Ray sensor works (min_dist={distances.min():.2f}, max_dist={distances.max():.2f})")
except Exception as e:
    print(f"✗ Sensor test failed: {e}")
    sys.exit(1)

# Test 5: Test obstacle avoidance controller
print("\n5. Testing obstacle avoidance controller...")
try:
    avoid_controller = ObstacleAvoidanceController(
        d_safe=1.0, k_avoid=0.15, a_avoid_max=0.5
    )
    action = avoid_controller.compute_avoidance_action(distances, directions, normals)
    print(f"✓ Avoidance controller works (action_norm={np.linalg.norm(action):.3f})")
    
    # Test velocity projection
    velocity = np.array([1.0, 0.0])
    projected_vel = avoid_controller.project_velocity(velocity, distances, directions, normals)
    print(f"✓ Velocity projection works")
except Exception as e:
    print(f"✗ Avoidance controller test failed: {e}")
    sys.exit(1)

# Test 6: Test formation controller
print("\n6. Testing formation controller...")
try:
    form_controller = FormationPDController(k_p=0.5, k_d=0.3)
    action = form_controller.compute_formation_action(
        np.array([1.0, 0.0]), np.array([0.1, 0.0]),
        np.array([0.0, 0.0]), np.array([-1.0, 0.0])
    )
    print(f"✓ Formation controller works (action_norm={np.linalg.norm(action):.3f})")
except Exception as e:
    print(f"✗ Formation controller test failed: {e}")
    sys.exit(1)

# Test 7: Create and test environment (Stage 1)
print("\n7. Testing Stage 1 environment (Leader training)...")
try:
    env = LeaderFollowerNavEnv(config, training_stage=1)
    obs, info = env.reset()
    print(f"✓ Environment created and reset")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    print(f"✓ Environment step works (reward={reward:.3f})")
    env.close()
except Exception as e:
    print(f"✗ Stage 1 environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test Stage 2 environment
print("\n8. Testing Stage 2 environment (Follower training)...")
try:
    env = LeaderFollowerNavEnv(config, training_stage=2)
    obs, info = env.reset()
    print(f"✓ Environment created and reset")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    print(f"✓ Environment step works (reward={reward:.3f})")
    env.close()
except Exception as e:
    print(f"✗ Stage 2 environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Test policy network imports
print("\n9. Testing neural network models...")
try:
    import torch
    from lf_mha_multi.models.leader_mha import LeaderMHAFeaturesExtractor
    from lf_mha_multi.models.follower_policy import FollowerFeaturesExtractor
    print("✓ Neural network models imported successfully")
except Exception as e:
    print(f"✗ Neural network test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe system is ready to use!")
print("\nNext steps:")
print("  1. Train leader: python train_leader.py --total-timesteps 10000")
print("  2. Train follower: python train_follower.py --total-timesteps 10000")
print("  3. Evaluate: python eval.py --n-episodes 10")

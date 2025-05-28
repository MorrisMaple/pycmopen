from pycmo.lib.actions import set_unit_heading_and_speed, manual_attack_contact, delete_unit, add_unit, set_unit_to_mission
from pycmo.agents.base_agent import BaseAgent
from pycmo.lib.features import Multi_Side_FeaturesFromSteam, Unit
from pycmo.lib.logger import Logger
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import random
import logging

# 只 import DQN_Agent
from module.batch_agent.DQN_agent import DQN_Agent

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

class MyAgent(BaseAgent):
    class EnemyInfo:
        def __init__(self, player_side, enemy_side):
            self.player_side = player_side
            self.enemy_side = enemy_side
            self.enemy_alive = {}
            self.initial_enemy_count = 0
            self.enemy_alive_count = 0
            self.prev_enemy_alive_count = 0
            self.order = []
        def init_episode(self, features):
            self.enemy_alive = {u.Name: 1 for u in features.units[self.enemy_side]}
            self.enemy_alive_count = len(self.enemy_alive)
            self.prev_enemy_alive_count = len(self.enemy_alive)
            if not self.order:
                self.initial_enemy_count = len(self.enemy_alive)
                self.order = [u.Name for u in features.units[self.enemy_side]]
        def get_enemy_found(self, features):
            return 1 if len(features.contacts[self.player_side]) > 0 else 0
        def update_alive(self, features):
            current_ids = {u.Name for u in features.units[self.enemy_side]}
            for name, alive in list(self.enemy_alive.items()):
                if alive == 1 and name not in current_ids:
                    self.enemy_alive[name] = 0
            for name in current_ids:
                if name not in self.enemy_alive:
                    self.enemy_alive[name] = 1
            self.enemy_alive_count = sum(self.enemy_alive.values())
        def alive_ratio(self):
            return (sum(self.enemy_alive.values()) / self.initial_enemy_count) if self.initial_enemy_count > 0 else 0.0

    class FriendlyInfo:
        def __init__(self, player_side):
            self.player_side = player_side
            self.order = []
            self.alive = {}
        def init_episode(self, features):
            if not self.order:
                all_names = [u.Name for u in features.units[self.player_side]]
                for name in all_names:
                    if name not in self.order:
                        self.order.append(name)
            self.alive = {name: 1 for name in self.order}
        def update_alive(self, features):
            current_ids = {u.Name for u in features.units[self.player_side]}
            for name, alive in list(self.alive.items()):
                if alive == 1 and name not in current_ids:
                    self.alive[name] = 0
                elif alive == 0 and name in current_ids:
                    self.alive[name] = 1
        def alive_mask(self):
            return [self.alive.get(n, 0) for n in self.order]

    def __init__(self, player_side: str, enemy_side: str = None):
        super().__init__(player_side)
        self.player_side = player_side
        self.enemy_side = enemy_side
        self.logger = logging.getLogger(f"MyAgent")
        self.logger.setLevel(logging.INFO)

        # 只保留 DQN 參數
        class Args:
            def __init__(self):
                self.hidden_dim = 64
                self.n_agents = 3
                self.enemy_num = 3
                self.input_size = 7 + 5 * (self.n_agents-1) + 4 * self.enemy_num
                self.n_actions = 4
        self.args = Args()
        self.input_size = self.args.input_size
        self.n_agents = self.args.n_agents
        self.n_actions = self.args.n_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用設備: {self.device}")

        # DQN agent
        self.dqn_agent = DQN_Agent(self.input_size, self.args).to(self.device)
        self.target_dqn_agent = DQN_Agent(self.input_size, self.args).to(self.device)
        self.target_dqn_agent.load_state_dict(self.dqn_agent.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn_agent.parameters(), lr=5e-4)
        self.gamma = 0.99

        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 32
        self.update_freq = 1000
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay_steps = 100000
        self.epsilon = self.eps_start
        self.total_steps = 0

        self.done_condition = 0.1
        self.max_distance = 90.0
        self.win_reward = 150
        self.min_win_reward = 50
        self.reward_scale = 25

        self.episode_init = True
        self.episode_step = 0
        self.episode_count = 0
        self.max_episode_steps = 500
        self.min_episode_steps = 200
        self.episode_done = False
        self.episode_reward = 0
        self.done = False
        self.prev_score = 0

        self.enemy_info = MyAgent.EnemyInfo(self.player_side, self.enemy_side)
        self.friendly_info = MyAgent.FriendlyInfo(self.player_side)

    def get_unit_info_from_observation(self, features: Multi_Side_FeaturesFromSteam, side: str, unit_name: str) -> Unit:
        units = features.units[side]
        for unit in units:
            if unit.Name == unit_name:
                return unit
        return None

    def get_done(self, state: list[np.ndarray]):
        if self.episode_step == 0:
            return False
        if self.episode_step >= self.max_episode_steps:
            return True
        done = True
        for i, name in enumerate(self.friendly_info.order):
            if state[i][0] > self.done_condition: 
                done = False
        return done

    def action(self, features: Multi_Side_FeaturesFromSteam) -> str:
        if self.episode_init:
            self.enemy_info.init_episode(features)
            self.friendly_info.init_episode(features)
            self.episode_init = False
            self.episode_count += 1
            self.episode_step = 0
            self.episode_reward = 0

        if features.sides_[self.player_side].TotalScore == 0:
            self.prev_score = 0

        has_unit = len(features.units[self.player_side]) > 0
        if not has_unit:
            self.logger.warning(f"找不到任何單位")
            return self.reset()

        current_state = self.get_states(features)  # List[np.ndarray] for all agents

        # 若有前一步資料則存進 replay buffer
        if hasattr(self, "prev_state") and hasattr(self, "prev_action") and self.episode_step > 0:
            rewards = self.get_rewards(features, self.prev_state, current_state, features.sides_[self.player_side].TotalScore)
            done = self.get_done(current_state)
            for agent_idx in range(self.n_agents):
                # 多 agent：每人一筆資料
                self.replay_buffer.push(self.prev_state[agent_idx], self.prev_action[agent_idx], rewards[agent_idx], current_state[agent_idx], done)
            self.episode_reward += np.mean(rewards)
            if done:
                self.logger.info(f"Episode {self.episode_count} done! Total Reward: {self.episode_reward:.2f}")
                return self.reset()

        # --- DQN動作選擇 ---
        state_tensor = torch.tensor(np.stack(current_state), dtype=torch.float32, device=self.device)  # [n_agents, feat]
        state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # [T=1, B=1, A, feat]
        with torch.no_grad():
            q_values = self.dqn_agent(state_tensor)[0, 0]  # [A, n_actions]
        has_enemy = len(features.contacts[self.player_side]) > 0

        masks = []
        for i in range(self.n_agents):
            mask = torch.ones(self.n_actions, dtype=torch.bool, device=self.device)
            if not has_enemy:
                mask[3] = False
            mount_ratio = current_state[i][5]
            if mount_ratio <= 0:
                mask[3] = False
            masks.append(mask)

        actions = []
        for ai in range(self.n_agents):
            q_agent = q_values[ai]
            mask = masks[ai]
            if random.random() < self.epsilon:
                allowed = mask.nonzero().squeeze(-1).tolist()
                act = random.choice(allowed) if allowed else 0
            else:
                q_agent_masked = q_agent.clone()
                q_agent_masked[~mask] = -float('inf')
                act = int(q_agent_masked.argmax().item())
            actions.append(act)

        # 執行動作
        self.friendly_info.update_alive(features)
        alive_mask = self.friendly_info.alive_mask()
        action_cmd = ""
        for idx, name in enumerate(self.friendly_info.order):
            if not alive_mask[idx]:
                continue
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if current_state[idx][0] < self.done_condition:
                action_cmd += "\n" + set_unit_heading_and_speed(
                    side=self.player_side,
                    unit_name=name,
                    heading=unit.CH,
                    speed=0
                )
            else:
                action_cmd += "\n" + self.apply_action(actions[idx], unit, features)

        if self.episode_step < 10:
            for unit in features.units[self.enemy_side]:
                action_cmd += "\n" + set_unit_to_mission(
                    unit_name=unit.Name,
                    mission_name='Kinmen patrol'
                )

        # 訓練 DQN
        if len(self.replay_buffer) >= self.batch_size:
            self.train()

        # 更新 epsilon
        self.total_steps += 1
        self.episode_step += 1
        self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * self.total_steps / self.eps_decay_steps
        self.epsilon = max(self.eps_end, self.epsilon)

        # target network
        if self.total_steps % self.update_freq == 0:
            self.target_dqn_agent.load_state_dict(self.dqn_agent.state_dict())

        self.prev_state = current_state
        self.prev_action = actions

        return action_cmd

    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)  # [B, feat]
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)    # [B]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device) # [B]
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)  # [B, feat]
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)     # [B]

        # shape to [T=1,B,A,feat]
        B = states.shape[0]
        A = self.n_agents
        states = states.view(B, self.input_size)
        next_states = next_states.view(B, self.input_size)

        # DQN forward
        q_values = self.dqn_agent(states.unsqueeze(0).unsqueeze(0))[0, 0]  # [B, n_actions]
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_dqn_agent(next_states.unsqueeze(0).unsqueeze(0))[0, 0]  # [B, n_actions]
            next_state_values = next_q_values.max(1)[0]
        expected_state_action_values = rewards + self.gamma * next_state_values * (1 - dones)
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_agent.parameters(), max_norm=5)
        self.optimizer.step()
        return loss.item()

    def get_state(self, features: Multi_Side_FeaturesFromSteam, ac: Unit) -> np.ndarray:
        """
        獲取當前狀態向量，包含自身單位和目標的相對位置資訊。
        :return: numpy 陣列，例如 
        [相對X, 相對Y, 敵人是否存在, 敵人存活比率,所有敵人位置]
        """
        target_lon = float(118.27954108343)
        target_lat = float(24.333113806906)
        max_distance = self.max_distance
        
        # 計算相對位置
        ac_lon = float(ac.Lon)
        ac_lat = float(ac.Lat)
        
        # 計算相對座標 (X,Y)，將經緯度差轉換為大致的平面座標
        # 注意：這是簡化的轉換，對於小範圍有效
        # X正方向為東，Y正方向為北
        earth_radius = 6371  # 地球半徑（公里）
        lon_scale = np.cos(np.radians(ac_lat))  # 經度在當前緯度的縮放因子
        
        # 1. 計算目標相對 X 和 Y（公里）
        dx = (target_lon - ac_lon) * np.pi * earth_radius * lon_scale / 180.0
        dy = (target_lat - ac_lat) * np.pi * earth_radius / 180.0
        dist = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / max_distance
        dy_norm = dy / max_distance
        dist_norm = dist / max_distance

        # 計算目標方位角（0=東，逆時針為正）
        target_angle = np.arctan2(dy, dx)

        # CMO 的 CH: 0=北，順時針增
        # 轉到 0=東，逆時針增：heading_math = 90°−CH
        heading_rad = np.deg2rad(90.0 - ac.CH)

        # 相對角度 = 目標方位 − 自身航向
        relative_angle = target_angle - heading_rad

        # 正規化到 [-π, π]
        relative_angle = (relative_angle + np.pi) % (2*np.pi) - np.pi

        # 如需用 sin/cos 表示
        relative_sin = np.sin(relative_angle)
        relative_cos = np.cos(relative_angle)

        # 敵方資訊處理: 檢查是否有敵人 & 更新存活狀態
        enemy_found = self.enemy_info.get_enemy_found(features)
        self.enemy_info.update_alive(features)
        alive_ratio = self.enemy_info.alive_ratio()
        step_ratio = self.episode_step / self.max_episode_steps

        # 計算彈藥持有比率
        mount_ratio = 0.0
        mounts = getattr(ac, 'Mounts', None)
        if mounts:
            for mount in mounts:
                name = getattr(mount, 'Name', None)
                weapons = getattr(mount, 'Weapons', [])
                if not weapons:
                    continue
                curr = weapons[0].QuantRemaining
                maxq = weapons[0].MaxQuant
                ratio = curr / maxq if maxq > 0 else 0.0
                if name == 'Hsiung Feng II Quad':
                    mount_ratio += ratio
                elif name == 'Hsiung Feng III Quad':
                    mount_ratio += ratio
            mount_ratio /= 2

        # 構建基礎狀態向量[距離, 方位sin, 方位cos, 敵人是否存在, 敵人存活比率,彈藥比率,步數比率]
        base_state = np.array([
                                dist_norm, #0
                                relative_sin, #1
                                relative_cos, #2
                                enemy_found, #3
                                alive_ratio, #4
                                mount_ratio, #5
                                step_ratio #6
                                ], dtype=np.float32)

        state_vec = base_state
        # 友軍狀態向量
        friendly_positions = []
        for name in self.friendly_info.order:
            friendly_unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if friendly_unit is not None :
                if friendly_unit.Name == ac.Name:
                    continue
                # 計算相對位置
                friendly_alive = 1.0
                friendly_dx = (float(friendly_unit.Lon) - ac_lon) * np.pi * earth_radius * lon_scale / 180.0
                friendly_dy = (float(friendly_unit.Lat) - ac_lat) * np.pi * earth_radius / 180.0
                friendly_dx_norm = friendly_dx / max_distance
                friendly_dy_norm = friendly_dy / max_distance
                friendly_dist_norm = np.sqrt(friendly_dx_norm**2 + friendly_dy_norm**2)

                # 計算方位角
                friendly_angle = np.arctan2(friendly_dy, friendly_dx)
                friendly_relative_angle = friendly_angle - heading_rad
                friendly_relative_angle = (friendly_relative_angle + np.pi) % (2*np.pi) - np.pi
                friendly_relative_sin = np.sin(friendly_relative_angle)
                friendly_relative_cos = np.cos(friendly_relative_angle)
                # 計算彈藥持有比率
                mount_ratio = 0.0
                mounts = getattr(friendly_unit, 'Mounts', None)
                if mounts:
                    for mount in mounts:
                        name = getattr(mount, 'Name', None)
                        weapons = getattr(mount, 'Weapons', [])
                        if not weapons:
                            continue
                        curr = weapons[0].QuantRemaining
                        maxq = weapons[0].MaxQuant
                        ratio = curr / maxq if maxq > 0 else 0.0
                        if name == 'Hsiung Feng II Quad':
                            mount_ratio += ratio
                        elif name == 'Hsiung Feng III Quad':
                            mount_ratio += ratio
                mount_ratio /= 2
            else:
                friendly_alive = 0.0
                friendly_dist_norm = 0.0
                friendly_relative_sin = 0.0
                friendly_relative_cos = 0.0
                mount_ratio = 0.0
            # 構建友軍狀態向量[存活, 距離, 方位sin, 方位cos, 彈藥比率]
            friendly_positions += [
                                    friendly_alive,
                                    friendly_dist_norm,
                                    friendly_relative_sin,
                                    friendly_relative_cos,
                                    mount_ratio
                                ]
        state_vec = np.concatenate([state_vec, friendly_positions])


        # 敵人狀態向量
        enemy_positions = []
        for name in self.enemy_info.order:
            enemy_unit = self.get_unit_info_from_observation(features, self.enemy_side, name)
            if enemy_unit is not None:
                enemy_alive = 1.0
                enemy_dx = (float(enemy_unit.Lon) - ac_lon) * np.pi * earth_radius * lon_scale / 180.0
                enemy_dy = (float(enemy_unit.Lat) - ac_lat) * np.pi * earth_radius / 180.0
                enemy_dx_norm = enemy_dx / max_distance
                enemy_dy_norm = enemy_dy / max_distance
                enemy_dist_norm = np.sqrt(enemy_dx_norm**2 + enemy_dy_norm**2)
                # 計算方位角
                enemy_angle = np.arctan2(enemy_dy, enemy_dx)
                enemy_relative_angle = enemy_angle - heading_rad
                enemy_relative_angle = (enemy_relative_angle + np.pi) % (2*np.pi) - np.pi
                enemy_relative_sin = np.sin(enemy_relative_angle)
                enemy_relative_cos = np.cos(enemy_relative_angle)
            else:
                enemy_alive = 0.0
                enemy_dist_norm = 0.0
                enemy_relative_sin = 0.0
                enemy_relative_cos = 0.0
            # 構建敵人狀態向量[存活, 距離, 方位sin, 方位cos]
            enemy_positions += [
                                    enemy_alive,
                                    enemy_dist_norm,
                                    enemy_relative_sin,
                                    enemy_relative_cos
                                ]

        # 整合敵人位置
        state_vec = np.concatenate([state_vec, enemy_positions])
       
        # 整合最終狀態向量
        raw_state = state_vec

        # normalized_state = self.normalize_state(raw_state)
        # print("Name:", ac.Name)
        # print("raw_state:", raw_state)
        # print("normalized_state:", normalized_state)
        
        # 返回狀態向量
        return raw_state
    
    def get_states(self, features: Multi_Side_FeaturesFromSteam) -> list[np.ndarray]:
        states = []
        # 對每個初始友軍單位按順序生成狀態，死亡的單位回傳默認值
        for name in self.friendly_info.order:
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if unit is None:
                # 單位死亡或不存在，返回預設零state
                state = np.zeros(self.input_size, dtype=np.float32)
                # state = self.normalize_state(raw_state)
                # print(f"單位 {name} 死亡或不存在，返回預設零state: {state}")
            else:
                state = self.get_state(features, unit)
            states.append(state)
        return states
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def get_reward(self, state: np.ndarray, next_state: np.ndarray, score: int) -> np.ndarray:
        # 計算全局 reward
        reward = 0
        
        # 場景score
        current_score = score
        # 場景score變化
        score_change = current_score - self.prev_score
        self.prev_score = current_score
        # 場景score變化獎勵
        reward += score_change
        # 發現敵人獎勵
        if state[3] == 0.0 and next_state[3] == 1.0:
            reward += 10
        # 往敵方探索獎勵
        reward += 200 * (state[0] - next_state[0])
        # 少一個敵人+10
        # if self.enemy_info.enemy_alive_count < self.enemy_info.prev_enemy_alive_count:
        #     reward += 10 *(self.enemy_info.prev_enemy_alive_count - self.enemy_info.enemy_alive_count)
        # self.enemy_info.prev_enemy_alive_count = self.enemy_info.enemy_alive_count

        if state[0] >= self.done_condition and next_state[0] < self.done_condition:
            reward += 20

        if next_state[0] < self.done_condition and self.done:
            win_reward = self.win_reward * (1- (self.episode_step - self.min_episode_steps) / (self.max_episode_steps - self.min_episode_steps))
            win_reward = max(win_reward, self.min_win_reward)
            reward += win_reward

        # 原始獎勵
        raw_reward = reward
        # 獲勝獎勵200 + 敵軍總數 7 *擊殺獎勵 20 + 最大距離獎勵 200*7
        max_return = self.win_reward + self.enemy_info.initial_enemy_count * 20 +  100
        scaled_reward = raw_reward/(max_return/self.reward_scale)
        # self.logger.info(f"raw reward: {raw_reward:.4f}, scaled reward: {scaled_reward:.4f}")
        # 將標量 reward 擴展為多代理人向量
        # return raw_reward
        return scaled_reward
    
    def get_rewards(self,features: Multi_Side_FeaturesFromSteam, state: list[np.ndarray], next_state: list[np.ndarray], score: int) -> list[np.ndarray]:
        rewards = []
        # 對每個初始友軍單位按順序生成狀態，死亡的單位回傳默認值
        for i, name in enumerate(self.friendly_info.order):
            unit = self.get_unit_info_from_observation(features, self.player_side, name)
            if unit is None:
                # 單位死亡或不存在，給予0獎勵
                reward = 0
            else:
                reward = self.get_reward(state[i], next_state[i], score)
            # 無論單位是否存活，都添加對應獎勵，確保長度一致
            rewards.append(reward)
        return rewards
    def apply_action(self, action: int, ac: Unit, features: Multi_Side_FeaturesFromSteam) -> str:
        """將動作轉換為 CMO 命令"""
        lat, lon = float(ac.Lat), float(ac.Lon)
        if action == 0: #前進
            heading = ac.CH
        elif action == 1: #左轉
            heading = ac.CH-30
        elif action == 2: #右轉
            heading = ac.CH+30
        elif action == 3:  # 攻擊
            # 檢查是否有彈藥
            has_ammo = False
            enemy = random.choice(features.contacts[self.player_side])
            for mount in ac.Mounts:
                name = getattr(mount, 'Name', None)
                if name not in ('Hsiung Feng II Quad', 'Hsiung Feng III Quad'):
                    continue
                        
                weapons = getattr(mount, 'Weapons', [])
                if weapons and weapons[0].QuantRemaining > 0:
                    if name == 'Hsiung Feng III Quad':
                        weapon_id = 1133
                    elif name == 'Hsiung Feng II Quad':
                        weapon_id = 1934
                    has_ammo = True
                    break
            if not has_ammo:
                # 無彈藥，保持前進
                heading = ac.CH
            else:
                # 有彈藥，執行攻擊
                return manual_attack_contact(
                    attacker_id=ac.ID,
                    contact_id=enemy['ID'],
                    weapon_id=weapon_id,
                    qty=1
                )
            
        if heading > 360:
            heading = heading - 360
        elif heading < 0:
            heading = 360 + heading
        return set_unit_heading_and_speed(
            side=self.player_side,
            unit_name=ac.Name,
            heading=heading,
            speed=30
        )

    def reset(self):
        self.episode_init = True
        self.prev_state = None
        self.prev_action = None
        self.episode_step = 0
        self.episode_done = False
        self.episode_reward = 0
        self.done = False
        self.logger.info("重置遊戲狀態，準備開始新的episode")
        return ""

# Author: Minh Hua
# Date: 08/16/2021
# Last Update: 06/16/2022
# Purpose: A Command environment.

# imports
import collections, enum
import os
from time import sleep
import logging

from pycmo.lib.actions import AvailableFunctions
from pycmo.lib.features import Features, FeaturesFromSteam, Multi_Side_FeaturesFromSteam
from pycmo.lib.protocol import Client, SteamClient, SteamClientProps
from pycmo.configs.config import get_config
from pycmo.lib.tools import cmo_steam_observation_file_to_xml
from xml.parsers.expat import ExpatError

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from typing import Optional, Callable

class TimeStep(
    collections.namedtuple(
        "TimeStep", ["step_id", "step_type", "reward", "observation"]
    )
):
    """
    Description:
        Returned with every call to `step` and `reset` on an environment.
        
        A `TimeStep` contains the data emitted by an environment at each step of
        interaction. A `TimeStep` holds a `step_type`, an `observation`, and an
        associated `reward` and `discount`.
        
        The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
        `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
        have `StepType.MID.

    Attributes:
        step_id: A scalar that represents the ID of a step.
        step_type: A `StepType` enum value.
        reward: A scalar, or 0 if `step_type` is `StepType.FIRST`, i.e. at the start of a sequence.
        observation: A NumPy array, or a dict, list or tuple of arrays.

    Author:
        DeepMind
    """

    def first(self):
        return self.step_type is StepType.FIRST

    def mid(self):
        return self.step_type is StepType.MID

    def last(self):
        return self.step_type is StepType.LAST

class StepType(enum.IntEnum):
    """
    Description:
        Defines the status of a `TimeStep` within a sequence.

    Author:
        DeepMind
    """

    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1
    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

class CPEEnv():
    """
    A wrapper that extracts observations from and sends actions to Command: Professional Edition.
    """
    def __init__(self, step_dest: str, step_size: list, player_side: str, scen_ended_path: str) -> None:
        """
        Description:
            Initializes the environment for one session.

        Keyword Arguments:
            step_dest: the path of the folder to hold the steps.
            step_size: a list containing the step size in the format ["h", "m", "s"].
            player_side: the string identifying the player's side. Side should exist in the game.
            scen_ended_path: the path to the text file that records whether a scenario has ended or not.

        Returns:
            None
        """
        self.client = Client() # initialize a client to send data to the game
        self.client.connect() # connect the client to the game
        self.player_side = player_side # the player's side, this is used to identify units that the player can actually control
        self.step_dest = step_dest # the path to the folder containing the xml steps files. These steps files are used to generate observations.
        self.scen_ended = scen_ended_path # the path to the text file recording whether or not the scenario has ended. "hacky" way to determine when a scenario ends because the current Lua command for this check is buggy in-game.
        # the step size in h, m, s
        self.h = step_size[0]
        self.m = step_size[1]
        self.s = step_size[2]

    def reset(self) -> TimeStep:
        """
        Description:
            Starts a new sequence and returns the first TimeStep of this sequence.

        Keyword Arguments:
            None
        
        Returns:
            (TimeStep) named tuple containing step_id, step_type, reward, observation.
        """
        f = open(self.scen_ended, 'w') # note in the scenario has ended file that the scenario has ended
        f.write('False')
        return TimeStep(0, StepType(0), 0, self.get_obs(0)) # return initial time step

    def step(self, cur_time, step_id, action=None) -> TimeStep:
        """
        Description:
            Updates the environment according to the action and returns a `TimeStep`. 

        Keyword Arguments:
            cur_time: the current time, used to check whether the game has finished progressing so that we can get the observation. Must be converted to UNIX format because the game uses C# ticks.
            step_id: the current step ID.
            action: a string containing the Lua script to send to the environment.
        
        Returns:
            (TimeStep) named tuple containing step_id, step_type, reward, observation. 
        """
        # send the agent's action
        if action != None:
            self.client.send(action)

        # step the environment forwards
        self.client.send("\nVP_RunForTimeAndHalt({Time='" + str(self.h) + ":" + str(self.m) + ":" + str(self.s) + "'})")

        # get the corresponding observation and reward
        # continuously poll the game until the correct time step duration has passed
        paused = False
        dur_in_secs = (int(self.h) * 3600) + (int(self.m) * 60) + int(self.s)
        step_file_name = str(step_id) + '.xml'
        while not (paused or self.check_game_ended()):
            data = "--script \nlocal now = ScenEdit_CurrentTime() \nlocal elapsed = now - {} \nif elapsed >= {} then \nfile = io.open('{}', 'w') \nio.output(file) \ntheXML = ScenEdit_ExportScenarioToXML()\nio.write(theXML) \nio.close(file) \nend".format(cur_time, dur_in_secs, self.step_dest + str(step_id) + '.xml')            
            self.client.send(data)
            if step_file_name in os.listdir(self.step_dest): # the game has been progressed and the new step information has been saved
                paused = True
                observation = Features(os.path.join(self.step_dest, step_file_name), self.player_side)
                reward = observation.side_.TotalScore
                return TimeStep(step_id, StepType(1), reward, observation)
            sleep(0.1) # else, sleep for 0.1 second to give the game a chance to catch up
        # if the game has ended, then save the timestep information with a different step type
        observation = self.get_obs(step_id)
        reward = observation.side_.TotalScore
        return TimeStep(step_id, StepType(2), reward, observation)        

    def get_obs(self, step_id:int) -> Features:
        """
        Description:
            Returns the observation at a particular timestep. This is done by calling the game to export the entire scenario to xml.

        Keyword Arguments:
            step_id: the index of the current step, starting at 0.
        
        Returns:
            (Features) named tuple containing the game observations at the current time index.
        """
        data = "--script \nfile = io.open('{}', 'w')".format(self.step_dest + str(step_id) + '.xml')
        data += "\nio.output(file) \ntheXML = ScenEdit_ExportScenarioToXML() \nio.write(theXML) \nio.close(file)"
        self.client.send(data)
        return Features(os.path.join(self.step_dest, str(step_id) + ".xml"), self.player_side)     

    def reset_connection(self) -> bool:
        """
        Description:
            Restart the client's connection the game.

        Keyword Arguments:
            None

        Returns:
            (bool) connection successful or not
        """
        return self.client.restart()
    
    def check_game_ended(self) -> bool:
        """
        Description:
            Check whether the scenario has ended.
            Reads the recorded txt file in the scenario has ended folder to check for this information.
            The in-game Lua check for this is currently buggy.
        
        Keyword Arguments:
            None

        Returns:
            (bool) whether the game has ended
        """
        data = "--script \nlocal scen = VP_GetScenario() \nif scen.CurrentTimeNum - scen.StartTimeNum >= scen.DurationNum then \nfile = io.open('{}', 'w') \nio.output(file) \nio.write('True') \nio.close(file) \nend".format(self.scen_ended)
        self.client.send(data)
        f = open(self.scen_ended, 'r')
        if f.readline() == 'True':
            return True
        return False
    
    def action_spec(self, observation:Features) -> AvailableFunctions:
        """
        Description:
            Returns the available actions given an observation.
        
        Keyword Arguments:
            observation: the current observations in the game.

        Returns:
            (AvailableFunctions) a list of possible actions
        """        
        return AvailableFunctions(observation)

    def close(self) -> bool:
        """
        Description:
            Close the client connection and the environment.
        
        Keyword Arguments:
            None

        Returns:
            (bool) whether the connection was successfully ended.    
        """
        return self.client.end_connection()
    
class CMOEnv():
    """
    A wrapper that extracts observations from and sends actions to Command: Modern Operations (Steam).
    """
    def __init__(self,
                 steam_client_props:SteamClientProps,
                 observation_path: str, 
                 action_path: str,
                 scen_ended_path: str,
                 pycmo_lua_lib_path: str | None = None,
                 player_side: str = None,
                 side_list: list[str] = None,
                 max_resets: int = 20):
        self.client = SteamClient(props=steam_client_props) # initialize a client to send data to the game
        if not self.client.connect(): # connect the client to the game
            raise FileNotFoundError("No running instance of Command to connect to.")
        
        
        self.player_side = player_side # the player's side, this is used to identify units that the player can actually control
        self.side_list = side_list #
        
        self.observation_path = observation_path # the path to the folder containing the xml steps files. These steps files are used to generate observations.
        self.action_path = action_path
        self.scen_ended = scen_ended_path # the path to the text file recording whether or not the scenario has ended. "hacky" way to determine when a scenario ends because the current Lua command for this check is buggy in-game.
        if not pycmo_lua_lib_path:
            config = get_config()
            pycmo_lua_lib_path = os.path.join(config['pycmo_path'], 'lua', 'pycmo_lib.lua')
        self.pycmo_lua_lib_path = pycmo_lua_lib_path # the path to the pycmo_lib.lua file
        self.max_resets = max_resets

        self.current_observation = None
        self.step_id = 0

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Watchdog 相關屬性
        self.observer = None
        self.inst_callback = None
        
        # 設定 Watchdog 監控
        self._setup_inst_watcher()
        
        self.inst_changed = False
        self.check_inst_changed(callback=self.on_inst_changed)

        self.obs = None
        # per comment (https://github.com/duyminh1998/pycmo/issues/25#issuecomment-1817773399) on issue #25, we need to edit the *_scen_has_ended.inst file when we init the env that the scenario has ended?
        with open(self.scen_ended, 'r') as file:
            data = file.readlines()
            data[7] = '  "Comments": "true",\n'
        with open(self.scen_ended, 'w') as file:
            file.writelines(data)
    
    def reset(self, close_scenario_end_and_player_eval_messages:bool=False) -> TimeStep:
        try:
            if close_scenario_end_and_player_eval_messages:
                self.client.close_scenario_end_and_player_eval_messages()

            restart_result = self.client.restart_scenario()

            # check that the scenario loaded event has fired correctly in CMO, and if not, restart the scenario
            retries = 0
            while (not restart_result or self.check_game_ended()) and retries < self.max_resets:
                self.logger.info(f"Scenario not loaded properly. Retrying... (Attempt {retries + 1} of {self.max_resets})")
                restart_result = self.client.restart_scenario()
                retries += 1
            if self.check_game_ended():
                raise ValueError("Scenario not restarting and loading properly. Please check game files.")

            self.client.send('')

            initial_observation = self.get_obs()
            if not initial_observation:
                raise FileNotFoundError("Cannot find observation file to reset the environment.")
            self.current_observation = initial_observation
            reward = 0 #initial_observation.side_.TotalScore
            self.step_id = 0
            # self.action_space = AvailableFunctions(features=self.current_observation)

            self.client.start_scenario()

            return TimeStep(self.step_id, StepType(0), reward, initial_observation) # return initial time step
        
        except FileNotFoundError:
            raise FileNotFoundError("Cannot find scen_has_ended.txt.")

    def send_action(self, action) -> None:
        if action != None: 
            try:
                self.client.send(action)
            except PermissionError:
                self.logger.debug("SteamClient was not able to write the agent's action. Stepping forwards with no new action.")
    
    def step(self, action=None) -> TimeStep:
        
        self.inst_changed = False
        # make sure the game is paused when step is called
        # while self.step_id > 0 and not self.check_game_ended(): ...
            # and not self.client.window_exists(window_name=self.client.scenario_paused_popup_name) \
            
        self.logger.debug(f"Step {self.step_id}")
        # send the agent's action
        if action != None: 
            try:
                self.client.send(action)
            except PermissionError:
                self.logger.debug("SteamClient was not able to write the agent's action. Stepping forwards with no new action.")
        
        # step the environment forwards
        #if not self.check_game_ended(): self.client.start_scenario() # step the game forwards until the message box appears
        self.logger.debug("2")
        while True:
            # self.logger.debug("3")
        # if the game has ended, then save the timestep information with a different step type
            if self.check_game_ended():
                observation = self.obs
                reward = 0#observation.side_.TotalScore
                # self.action_space.refresh(features=observation)
                return TimeStep(self.step_id, StepType(2), reward, observation)
            elif self.inst_changed:
                break
        self.logger.debug("4")
        new_observation = self.obs
        self.logger.debug("5")
        if new_observation.meta.Time == self.current_observation.meta.Time:
            self.logger.debug(f"Time is not advancing. Old time: {self.current_observation.meta.Time}, New time: {new_observation.meta.Time}. Moving forward with old state.")
            observation = self.current_observation
            reward = 0#self.current_observation.side_.TotalScore
            return TimeStep(self.step_id, StepType(1), reward, observation)            
        self.logger.debug("6")
        self.step_id += 1
        observation = new_observation
        reward = 0 #observation.side_.TotalScore
        new_timestep = TimeStep(self.step_id, StepType(1), reward, observation)
        self.logger.debug("7")
        self.current_observation = new_observation
        # self.action_space.refresh(features=self.current_observation)
        self.logger.debug("8")
        return new_timestep

    def get_obs(self) -> FeaturesFromSteam:
        get_obs_retries = 0
        max_get_obs_retries = 10000
        while True:
            try:
                if self.side_list:
                    obs = Multi_Side_FeaturesFromSteam(cmo_steam_observation_file_to_xml(self.observation_path), self.side_list)
                elif self.player_side:
                    obs = FeaturesFromSteam(cmo_steam_observation_file_to_xml(self.observation_path), self.player_side) 
                return obs
            except (TypeError, ExpatError):
                get_obs_retries += 1
                # print("get_obs_retries: ",get_obs_retries)
                sleep(0.0001)
                if get_obs_retries > max_get_obs_retries:
                    raise TimeoutError("CMOEnv unable to get observation.")
    
    def action_spec(self, observation:Features | FeaturesFromSteam) -> AvailableFunctions:    
        return AvailableFunctions(observation)

    def check_game_ended(self) -> bool:
        try:
            scenario_ended = cmo_steam_observation_file_to_xml(self.scen_ended)
            if scenario_ended == "true": # \
                # or self.client.window_exists(window_name=self.client.scenario_end_popup_name):
                return True
            return False
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find {self.scen_ended}")
        
    def end_game(self) -> TimeStep:
        self.logger.info(f"Ending game after {self.step_id} steps.")
        pycmo_lua_lib_path = self.pycmo_lua_lib_path.replace('\\', '/')
        export_observation_event_name = 'Export observation'
        action = f"ScenEdit_RunScript('{pycmo_lua_lib_path}', true)\nteardown_and_end_scenario('{export_observation_event_name}', true)"
        return self.step(action)


    class InstFileHandler(FileSystemEventHandler):
        def __init__(self, parent):
            self.parent = parent
            
        def on_modified(self, event):
            # 只處理特定的 observation_path 文件
            if event.src_path == self.parent.observation_path:
                # self.parent.logger.debug(f"檢測到觀察文件修改: {event.src_path}")
                if self.parent.inst_callback:
                    self.parent.inst_callback(event.src_path)

    def _setup_inst_watcher(self):
        """設定 Watchdog 監控特定的 .inst 文件"""
        try:
            if not os.path.exists(self.observation_path):
                raise FileNotFoundError(f"Observation file not found: {self.observation_path}")
                
            event_handler = self.InstFileHandler(self)
            self.observer = Observer()
            # 監控 observation_path 所在的目錄，但處理器會過濾特定文件
            watch_dir = os.path.dirname(self.observation_path)
            self.observer.schedule(event_handler, watch_dir, recursive=False)
            self.observer.start()
            self.logger.debug(f"Started watching specific .inst file: {self.observation_path}")
        except Exception as e:
            self.logger.error(f"Failed to setup inst watcher: {str(e)}")
            raise

    def check_inst_changed(self, callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Description:
            設定一個回調函數，在 .inst 文件改變時立即觸發。
            如果提供了回調函數，將其設置為觸發目標；否則僅啟動監控。
        
        Keyword Arguments:
            callback: 可選的回調函數，接受改變的文件路徑作為參數，在 .inst 文件改變時立即調用
        
        Returns:
            None
        """
        if callback:
            self.inst_callback = callback
            self.logger.debug("Inst change callback set")
        else:
            self.logger.debug("Inst change monitoring started without callback")

    def close(self) -> bool:
        """
        Description:
            Close the client connection and the environment, including stopping the watchdog observer.
        
        Returns:
            (bool) whether the connection was successfully ended.    
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.logger.debug("Watchdog observer stopped")
        return self.client.end_connection()
    
    def on_inst_changed(self, event_path: str) -> None:
        """
        Description:
            Callback function for when the .inst file changes.
        """
        # with open(event_path, 'r', encoding='utf-8') as file:
        #     file_content = file.read()
            
        # # 记录文件内容（可以只记录前100个字符作为摘要）
        # content_summary = file_content[130:300] + "..." if len(file_content) > 300 else file_content
        # self.logger.debug(f"修改后内容摘要: {content_summary}")
        # print("111")
        self.logger.debug("inst_changed")
        # print("222")
        new_obs = self.get_obs()
        # print("333")
        if self.obs is None:
            self.obs = new_obs
        elif new_obs.meta.Time != self.obs.meta.Time:
            self.inst_changed = True
            self.obs = new_obs
            self.logger.debug("obs changed")
        # print("444")

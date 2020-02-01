# -*- coding: UTF-8 -*-
###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015-16  Cambridge University Engineering Department 
# Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################
import time
from datetime import datetime

from utils.dact import DactItem

'''
SuccessEvaluator.py - module for determining objective and subjective dialogue success 
======================================================================================

Copyright CUED Dialogue Systems Group 2016

.. seealso:: PyDial Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`ontology.Ontology` |.|
    import :class:`evaluation.EvaluationManager.Evaluator` |.|

************************

'''
__author__ = "cued_dialogue_systems_group"
from EvaluationManager import Evaluator
from utils import Settings, ContextLogger, DiaAct, dact
from ontology import Ontology
import numpy as np
import copy

from model.vime import vime
from model.cme import cme
from model.scme import scme
import scipy.io as scio
import ConfigParser  

# import matplotlib as mpl
import os
# mpl.use('Agg')

import matplotlib.pyplot as plt

logger = ContextLogger.getLogger('')


class ObjectiveSuccessEvaluator(Evaluator):
    '''
    This class provides a reward model based on objective success. For simulated dialogues, the goal of the user simulator is compared with the the information the system has provided. 
    For dialogues with a task file, the task is compared to the information the system has provided. 
    '''
    
    def __init__(self, domainString):
        super(ObjectiveSuccessEvaluator, self).__init__(domainString)

        # only for nice prints
        self.evaluator_label = "objective success evaluator"
        self.evaluator_short_label = "suc"

        # DEFAULTS:
        self.reward_venue_recommended = 0  # we dont use this. 100
        self.penalise_all_turns = True   # We give -1 each turn. Note that this is done thru this boolean
        self.wrong_venue_penalty = 0   # we dont use this. 4
        self.not_mentioned_value_penalty = 0  # we dont use this. 4
        self.successReward = 20
        self.using_tasks = False
        self.failPenalty = 0
        self.user_goal = None
        # pw439 added
        self.delayed_turn_penalty_set_in = 0  # number of batch sizes after which turn penalty is used for learning
        self.turn_thresh = 0  # threshold, if not all turns are penalized
        self.pre_trg = False  # if pre-train data collection, set to True

        #improvement==================================
        self.intrinsic_reward_method = None
        self.conf = ConfigParser.ConfigParser()  
        self.num_actions = 40   #CR=16  SFR=25  LAP=40
        self.num_belief_states = 257  #CR=268  SFR=636  LAP=257

        # CONFIG:

        #if Settings.config.has_option('dqnpolicy', 'n_in'):
        #    self.num_belief_states = Settings.config.getint('dqnpolicy', 'n_in')
        if Settings.config.has_option('scme', 'method'):
            self.intrinsic_reward_method = Settings.config.get('scme', 'method')
        if Settings.config.has_option('scme', 'alpha'):
            self.alpha = Settings.config.getfloat('scme', 'alpha')
        if Settings.config.has_option('scme', 'beta'):
            self.beta = Settings.config.getfloat('scme', 'beta')
        if Settings.config.has_option('scme', 'annealing'):
            self.annealing = Settings.config.getfloat('scme', 'annealing')
        if Settings.config.has_option('scme', 'feat_size'):
            self.feat_size = Settings.config.getint('scme', 'feat_size')
        #improvement==================================

        if Settings.config.has_option('policy', 'inpolicyfile'):
            self.in_policy_file = Settings.config.get('policy', 'inpolicyfile')
        if Settings.config.has_option('policy', 'outpolicyfile'):
            self.out_policy_file = Settings.config.get('policy', 'outpolicyfile')
        if Settings.config.has_option('GENERAL', 'seed'):
            self.random_seed = Settings.config.getint('GENERAL', 'seed')
        if Settings.config.has_option('eval', 'pretrg'):
            self.pre_trg = Settings.config.getboolean('eval', 'pretrg')
            if self.pre_trg: print 'curiosity pre-trg data collection is running'
        if Settings.config.has_option('eval', 'curiosityreward'):
            self.curiosityreward = Settings.config.getboolean('eval', 'curiosityreward')
            if self.curiosityreward:
                self.feat_size = 200
                if Settings.config.has_option("eval", "feat_size"):
                    self.feat_size = Settings.config.getint("eval", "feat_size")
                self.rew_scale = 1.0
                if Settings.config.has_option("eval", "rew_scale"):
                    self.rew_scale = Settings.config.getfloat("eval", "rew_scale")
                self.model_name = 'trained_curiosityacer-shuffle22_feat200'
                if Settings.config.has_option("eval", "model_name"):
                    self.model_name = Settings.config.get("eval", "model_name")
        if Settings.config.has_option('eval', 'delayed_turn_penalty_set_in'):
            self.delayed_turn_penalty_set_in = Settings.config.getint('eval', 'delayed_turn_penalty_set_in')
        if Settings.config.has_option('eval', 'turn_thresh'):
            self.turn_thresh = Settings.config.getint('eval', 'turn_thresh')
        if Settings.config.has_option('eval', 'rewardvenuerecommended'):
            self.reward_venue_recommended = Settings.config.getint('eval', 'rewardvenuerecommended')
        if Settings.config.has_option('eval', 'penaliseallturns'):
            self.penalise_all_turns = Settings.config.getboolean('eval', 'penaliseallturns')
        if Settings.config.has_option('eval', 'wrongvenuepenalty'):
            self.wrong_venue_penalty = Settings.config.getint('eval', 'wrongvenuepenalty')
        if Settings.config.has_option('eval', 'notmentionedvaluepenalty'):
            self.not_mentioned_value_penalty = Settings.config.getint('eval', 'notmentionedvaluepenalty')
        if Settings.config.has_option("eval", "failpenalty"):
            self.failPenalty = Settings.config.getint("eval", "failpenalty")
        if Settings.config.has_option("eval", "successreward"):
            self.successReward = Settings.config.getint("eval", "successreward")
        if Settings.config.has_option("eval_"+domainString, "failpenalty"):
            self.failPenalty = Settings.config.getint("eval_"+domainString, "failpenalty")
        if Settings.config.has_option("eval_" + domainString, "successreward"):
            self.successReward = Settings.config.getint("eval_" + domainString, "successreward")
        if Settings.config.has_option("dialogueserver", "tasksfile"):
            self.using_tasks = True  # will record DM actions to deduce objective success against a given task:


        self.venue_recommended = False
        self.mentioned_values = {}  # {slot: set(values), ...}
        sys_reqestable_slots = Ontology.global_ontology.get_system_requestable_slots(self.domainString)
        for slot in sys_reqestable_slots:
            self.mentioned_values[slot] = set(['dontcare'])

                
        #improvement==================================
        #initial
        if self.intrinsic_reward_method == 'vime':
            self.vime_model = vime(self.num_belief_states, self.num_actions)
            self.vime_model.load_model('model/vime_model/' + self.in_policy_file)
            self.vime_reward = [] 
            self.old_parameter = []

        elif self.intrinsic_reward_method == 'cme':
            self.cme_model = cme(self.num_belief_states, self.num_actions)
            self.cme_model.load_model('model/cme_model/' + self.in_policy_file)
            self.cme_reward = [] 
            self.predstate = []
            self.origstate = []

        elif self.intrinsic_reward_method == 'scme':
            self.scme_model = scme(self.num_belief_states, self.num_actions)
            self.scme_model.load_model('model/scme_model/' + self.in_policy_file)
            self.scme_reward = []
            self.predzstate = []  
            self.origzstate = []  
            self.predstate = []
            self.origstate = []
        #improvement==================================

        self.DM_history = None

    def restart(self):
        """
        Initialise variables (i.e. start dialog with: success=False, venue recommended=False, and 'dontcare' as \
        the only mentioned value in each slot)

        :param: None
        :returns: None

        """
        super(ObjectiveSuccessEvaluator, self).restart()
        self.venue_recommended = False
        self.last_venue_recomended = None
        self.mentioned_values = {}      # {slot: set(values), ...}
        sys_reqestable_slots = Ontology.global_ontology.get_system_requestable_slots(self.domainString)
        for slot in sys_reqestable_slots:
            self.mentioned_values[slot] = set(['dontcare'])

        if self.using_tasks:
            self.DM_history = []

    def _getTurnReward(self, turnInfo):
        '''
        Computes the turn reward regarding turnInfo. The default turn reward is -1 unless otherwise computed.

        :param turnInfo: parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: int -- the turn reward.
        '''

        # Immediate reward for each turn.
        reward = -self.penalise_all_turns 

        # turn threshold option for turn penalty
        if self.num_turns < self.turn_thresh:
            reward = 0

        # delayed turn penalty set-in in learning
        if Settings.global_numiter <= self.delayed_turn_penalty_set_in:
            reward = 0

        if turnInfo is not None and isinstance(turnInfo, dict):

            if self.pre_trg:
                prev_state_vec = turnInfo['prev_state_vec']
                state_vec = turnInfo['state_vec']
                ac_1hot = turnInfo['ac_1hot']
                self.statelist.append(state_vec)
                self.prevstatelist.append(prev_state_vec)
                self.actionlist.append(np.where(ac_1hot == 1)[0][0])
                self.turnlist.append(turnInfo['turn_num'])


            #improvement==================================
            # #testing without intrinsic reward
            if self.in_policy_file[-1] == str(Settings.global_numiter)[-1]:   
                self.intrinsic_reward_method = None 
            
            #get reward    
            if self.intrinsic_reward_method == 'vime':
                old_parameters = self.vime_model.get_hyper_parameters()
                self.vime_model.load_model('model/vime_model/' + self.out_policy_file)
                new_parameters = self.vime_model.get_hyper_parameters()
                ri1 = self.vime_model.get_info_gain(new_parameters, old_parameters)
                reward += self.alpha * ri1   
                self.vime_reward.append(ri1)

            elif self.intrinsic_reward_method == 'cme':
                prev_state_vec = turnInfo['prev_state_vec']
                state_vec = turnInfo['state_vec']
                ac_1hot = turnInfo['ac_1hot']
                self.cme_model.load_model('model/cme_model/' + self.out_policy_file)
                ri1 = self.cme_model.reward(prev_state_vec, state_vec, ac_1hot)
                preds, actus = self.cme_model.predictedstate(prev_state_vec, state_vec, ac_1hot)
                reward += self.alpha * ri1
                self.cme_reward.append(ri1)
                self.predstate.append(preds)
                self.origstate.append(actus)
                
            elif self.intrinsic_reward_method == 'scme':
                prev_state_vec = turnInfo['prev_state_vec']
                state_vec = turnInfo['state_vec']
                ac_1hot = turnInfo['ac_1hot']
                self.scme_model.load_model('model/scme_model/' + self.out_policy_file)
                ri1 = (self.alpha - self.annealing * Settings.global_numiter) * self.scme_model.reward1(prev_state_vec, state_vec, ac_1hot)
                ri2 = self.beta * self.scme_model.reward2(prev_state_vec, state_vec, ac_1hot)
                predzs, origzs, preds, origs = self.scme_model.latent_code(prev_state_vec, state_vec, ac_1hot)
                ri = ri1 + ri2
                reward += ri
                self.scme_reward.append(ri)
                self.predzstate.append(predzs) 
                self.origzstate.append(origzs)     
                self.predstate.append(preds)
                self.origstate.append(origs)
            #improvement==================================



            if 'usermodel' in turnInfo and 'sys_act' in turnInfo:
                um = turnInfo['usermodel']
                self.user_goal = um.goal.constraints

                # unpack input user model um.
                # prev_consts = um.prev_goal.constraints
                prev_consts = copy.deepcopy(um.goal.constraints)
                for item in prev_consts:
                    if item.slot == 'name' and item.op == '=':
                        item.val = 'dontcare'
                requests = um.goal.requests
                sys_act = DiaAct.DiaAct(turnInfo['sys_act'])
                user_act = um.lastUserAct

                # Check if the most recent venue satisfies constraints.
                name = sys_act.get_value('name', negate=False)
                lvr = self.last_venue_recomended if hasattr(self, 'last_venue_recomended') else 'not existing'
                if name not in ['none', None]:
                    # Venue is recommended.
                    # possible_entities = Ontology.global_ontology.entity_by_features(self.domainString, constraints=prev_consts)
                    # is_valid_venue = name in [e['name'] for e in possible_entities]
                    self.last_venue_recomended = name
                    is_valid_venue = self._isValidVenue(name, prev_consts)
                    if is_valid_venue:
                        # Success except if the next user action is reqalts.
                        if user_act.act != 'reqalts':
                            logger.debug('Correct venue is recommended.')
                            self.venue_recommended = True   # Correct venue is recommended.
                        else:
                            logger.debug('Correct venue is recommended but the user has changed his mind.')
                    else:
                        # Previous venue did not match.
                        logger.debug('Venue is not correct.')
                        self.venue_recommended = False
                        logger.debug('Goal constraints: {}'.format(prev_consts))
                        reward -= self.wrong_venue_penalty

                # If system inform(name=none) but it was not right decision based on wrong values.
                if name == 'none' and sys_act.has_conflicting_value(prev_consts):
                    reward -= self.wrong_venue_penalty

                # Check if the system used slot values previously not mentioned for 'select' and 'confirm'.
                not_mentioned = False
                if sys_act.act in ['select', 'confirm']:
                    for slot in Ontology.global_ontology.get_system_requestable_slots(self.domainString):
                        values = set(sys_act.get_values(slot))
                        if len(values - self.mentioned_values[slot]) > 0:
                            # System used values which are not previously mentioned.
                            not_mentioned = True
                            break

                if not_mentioned:
                    reward -= self.not_mentioned_value_penalty

                # If the correct venue has been recommended and all requested slots are filled,
                # check if this dialogue is successful.
                if self.venue_recommended and None not in requests.values():
                    reward += self.reward_venue_recommended

                # Update mentioned values.
                self._update_mentioned_value(sys_act)
                self._update_mentioned_value(user_act)
            if 'sys_act' in turnInfo and self.using_tasks:
                self.DM_history.append(turnInfo['sys_act'])

        return reward

    def _isValidVenue(self, name, constraints):
        constraints2 = None
        if isinstance(constraints, list):
            constraints2 = copy.deepcopy(constraints)
            for const in constraints2:
                if const.slot == 'name':
                    if const.op == '!=':
                        if name == const.val and const.val != 'dontcare':
                            return False
                        else:
                            constraints2.remove(const)
                    elif const.op == '=':
                        if name != const.val and const.val != 'dontcare':
                            return False
            constraints2.append(DactItem('name','=',name))
        elif isinstance(constraints, dict): # should never be the case, um has DActItems as constraints
            constraints2 = copy.deepcopy(constraints)
            for slot in constraints:
                if slot == 'name' and name != constraints[slot]:
                    return False
            constraints2['name'] = name
        entities = Ontology.global_ontology.entity_by_features(self.domainString, constraints2)

#         is_valid_list = []
#         for ent in entities:
#             is_valid = True
#             for const in constraints:
#                 if const.op == '=':
#                     if const.val != ent[const.slot] and const.val != 'dontcare':
#                         is_valid = False
#                 elif const.op == '!=':
#                     if const.val == ent[const.slot]:
#                         is_valid = False
#             is_valid_list.append(is_valid)

        return any(entities)

    def _getFinalReward(self, finalInfo):
        '''
        Computes the final reward using finalInfo. Should be overridden by sub-class if values others than 0 should be returned.

        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: int -- the final reward, default 0.
        '''
        if finalInfo is not None and isinstance(finalInfo, dict):
            
            if 'usermodel' in finalInfo: # from user simulator
                um = finalInfo['usermodel']
                if um is None:
                    self.outcome = False
                elif self.domainString not in um:
                    self.outcome = False
                else:
                    requests = um[self.domainString].goal.requests
                    '''if self.last_venue_recomended is None:
                        logger.dial('Fail :( User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))
                    else:
                        if self.venue_recommended and None not in requests.values():
                            self.outcome = True
                            logger.dial('Success! User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))
                        else:
                            logger.dial('Fail :( User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))'''
                    if None not in requests.values():
                        valid_venue = self._isValidVenue(requests['name'], self.user_goal)
                        if valid_venue:
                            self.outcome = True
                            logger.dial(
                                'Success! User requests: {}'.format(requests))
                        else:
                            logger.dial(
                                'Fail :( User requests: {}'.format(requests))
                    else:
                        logger.dial(
                            'Fail :( User requests: {}'.format(requests))
            elif 'task' in finalInfo: # dialogue server with tasks
                task = finalInfo['task']
                if self.DM_history is not None:
                    informs = self._get_informs_against_each_entity()
                    if informs is not None:
                        for ent in informs.keys():
                            if task is None:
                                self.outcome = True   # since there are no goals, lets go with this ...
                            elif self.domainString not in task:
                                logger.warning("This task doesn't contain the domain: %s" % self.domainString)
                                logger.debug("task was: " + str(task))  # note the way tasks currently are, we dont have
                                # the task_id at this point ...
                                self.outcome = True   # This is arbitary, since there are no goals ... lets say true?
                            elif ent in str(task[self.domainString]["Ents"]):
                                # compare what was informed() against what was required by task:
                                required = str(task[self.domainString]["Reqs"]).split(",")
                                self.outcome = True
                                for req in required:
                                    if req.strip(" ") == 'name':
                                        continue
                                    if req.strip(" ") not in ','.join(informs[ent]):
                                        self.outcome = False
        
        return self.outcome * self.successReward - (not self.outcome) * self.failPenalty

    def _get_informs_against_each_entity(self):
        if len(self.DM_history) == 0:
            return None
        informs = {}
        currentEnt = None
        for act in self.DM_history:
            if 'inform(' in act:
                details = act.split("(")[1].split(",")
                details[-1] = details[-1][0:-1]  # remove the closing ) remove clothing
                if not len(details):
                    continue
                if "name=" in act:
                    for detail in details:
                        if "name=" in detail:
                            currentEnt = detail.split("=")[1].strip('"')
                            details.remove(detail)
                            break  # assumes only 1 name= in act -- seems solid assumption

                    if currentEnt in informs.keys():
                        informs[currentEnt] += details
                    else:
                        informs[currentEnt] = details
                elif currentEnt is None:
                    logger.warning("Shouldn't be possible to first encounter an inform() act without a name in it")
                else:
                    logger.warning('assuming inform() that does not mention a name refers to last entity mentioned')
                    informs[currentEnt] += details
        return informs

    def _update_mentioned_value(self, act):
        # internal, called by :func:`RewardComputer.get_reward` for both sys and user acts to update values mentioned in dialog
        #
        # :param act: sys or user dialog act
        # :type act: :class:`DiaAct.DiaAct`
        # :return: None

        sys_requestable_slots = Ontology.global_ontology.get_system_requestable_slots(self.domainString)
        for item in act.items:
            if item.slot in sys_requestable_slots and item.val not in [None, '**NONE**', 'none']:
                self.mentioned_values[item.slot].add(item.val)

    def _getResultString(self, outcomes):
        num_dialogs = len(outcomes)
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)


        
        #improvement==================================
        #save result  
        if self.intrinsic_reward_method == 'vime':
            rew = self.vime_reward[-1000:]
            save_fn = 'scme_plot/' + self.intrinsic_reward_method + '/' + self.out_policy_file + '.mat'
            scio.savemat(save_fn, {'rew': rew})

        elif self.intrinsic_reward_method == 'cme':
            rew = self.cme_reward[-1000:]
            pred_s = np.reshape(self.predstate, (-1, self.num_belief_states))
            orig_s = np.reshape(self.origstate, (-1, self.num_belief_states))
            pred_s = pred_s[-1000:,:]
            orig_s =  orig_s[-1000:,:]
            save_fn = 'scme_plot/' + self.intrinsic_reward_method + '/' + self.out_policy_file + '.mat'
            scio.savemat(save_fn, {'pred_s': pred_s, 'orig_s': orig_s, 'rew': rew})

        elif self.intrinsic_reward_method == 'scme':
            rew = self.scme_reward[-1000:]
            pred_zs = np.reshape(self.predzstate, (-1, 16))
            orig_zs = np.reshape(self.origzstate, (-1, 16))
            pred_s = np.reshape(self.predstate, (-1, self.num_belief_states))
            orig_s = np.reshape(self.origstate, (-1, self.num_belief_states))
            pred_zs = pred_zs[-1000:,:]
            orig_zs = orig_zs[-1000:,:]
            pred_s = pred_s[-1000:,:]
            orig_s =  orig_s[-1000:,:]
            save_fn = 'scme_plot/' + self.intrinsic_reward_method + '/' + self.out_policy_file + '.mat'
            scio.savemat(save_fn, {'pred_zs': pred_zs, 'orig_zs': orig_zs, 'pred_s': pred_s, 'orig_s': orig_s, 'rew': rew}) 
        #improvement==================================

            
        if self.pre_trg:
            date_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            if not os.path.exists('_curiosity_model/pretrg_data/'):
                os.mkdir('_curiosity_model/pretrg_data/')
            with open('_curiosity_model/pretrg_data/turn_action_'+date_time, 'w') as f:  # todo: include policy, env and seed in name
                # data collection for pretraining
                for num in range(len(self.turnlist)):
                    f.write('turn: '+str(self.turnlist[num])+' sys_action: '+str(self.actionlist[num]) +
                            '\n') #+str(self.statelist[num][0])+' prev_state: '+str(self.statelist[num][1]))
            with open('_curiosity_model/pretrg_data/state_'+date_time, 'w') as f2:
                np.savetxt(f2, np.array(self.statelist))
            with open('_curiosity_model/pretrg_data/prev_state' + date_time, 'w') as f3:
                np.savetxt(f3, np.array(self.prevstatelist))
            # reset lists
            self.turnlist = []
            self.actionlist = []
            self.statelist = []
            self.prevstatelist = []

        return 'Average success = {0:0.2f} +- {1:0.2f}'.format(100 * np.mean(outcomes),
                                                        100 * tinv * np.std(outcomes) / np.sqrt(num_dialogs))


class Sys2TextSuccessEvaluator(Evaluator):
    def __init__(self):
        self.goal = {}
        self.rewards = []
        self.traceDialog = 0
        domainString = 'CamRestaurants'
        self.evaluator_label = 0
        self.total_reward = 0
        self.outcome = False
        self.num_turns = 0
        super(Sys2TextSuccessEvaluator, self).__init__(domainString)

    def restart(self):
        self.outcome = False
        self.num_turns = 0
        self.total_reward = 0

    def _getTurnReward(self, turnInfo):
        '''
        Computes the turn reward which is always -1 if activated.

        :param turnInfo: NOT USED parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: int -- the turn reward.
        '''

        # Immediate reward for each turn.
        reward = -1 # should be - penalise all turns
        return reward

    def _getFinalReward(self, finalInfo):
        requests_fullfilled = finalInfo['usermodel'][self.domainString].reqFullfilled()
        accepted_venue = finalInfo['usermodel']['CamRestaurants'].getCurrentVenue()
        constraints = finalInfo['usermodel']['CamRestaurants'].getConstraintDict()
        constraints['name'] = accepted_venue
        entities = Ontology.global_ontology.entity_by_features(self.domainString, constraints)
        '''try:
            venue_constrains = Ontology.global_ontology.entity_by_features(self.domainString, {'name': accepted_venue})[0]
            print len(entities), requests_fullfilled, accepted_venue, constraints, venue_constrains['food'], venue_constrains['area'], venue_constrains['pricerange']
        except:
            print 'nothing', requests_fullfilled, accepted_venue, constraints
            pass'''
        if len(entities) > 0 and requests_fullfilled:
            self.outcome = True
        if self.outcome:
            return 20
        else:
            return 0


class SubjectiveSuccessEvaluator(Evaluator):
    '''
    This class implements a reward model based on subjective success which is only possible during voice interaction through the :mod:`DialogueServer`. The subjective feedback is collected and
    passed on to this class.
    '''
    
    def __init__(self, domainString):
        super(SubjectiveSuccessEvaluator, self).__init__(domainString)
        
        # only for nice prints
        self.evaluator_label = "subjective success evaluator"
               
        # DEFAULTS:
        self.penalise_all_turns = True   # We give -1 each turn. Note that this is done thru this boolean
        self.successReward = 20
        
        # CONFIG: t
        if Settings.config.has_option('eval', 'penaliseallturns'):
            self.penalise_all_turns = Settings.config.getboolean('eval', 'penaliseallturns')
        if Settings.config.has_option("eval", "successreward"):
            self.successReward = Settings.config.getint("eval", "successreward")
        if Settings.config.has_option("eval_" + domainString, "successreward"):
            self.successReward = Settings.config.getint("eval_" + domainString, "successreward")

    def restart(self):
        """
        Calls restart of parent.
    
        :param: None
        :returns: None
        """
        super(SubjectiveSuccessEvaluator, self).restart()
        
    def _getTurnReward(self,turnInfo):
        '''
        Computes the turn reward which is always -1 if activated. 
        
        :param turnInfo: NOT USED parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: int -- the turn reward.
        '''
        
        # Immediate reward for each turn.
        return -self.penalise_all_turns
        
    def _getFinalReward(self, finalInfo):
        '''
        Computes the final reward using finalInfo's field "subjectiveSuccess".
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: int -- the final reward, default 0.
        '''
        if finalInfo is not None and isinstance(finalInfo, dict):
            if 'subjectiveSuccess' in finalInfo:
                self.outcome = finalInfo['subjectiveSuccess']
                
        if self.outcome is None:
            self.outcome = 0;

        return self.outcome * self.successReward
    
    def _getResultString(self, outcomes):
        num_dialogs = len(outcomes)
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)
        return 'Average subj success = {0:0.2f} +- {1:0.2f}'.format(100 * np.mean(outcomes), \
                                                            100 * tinv * np.std(outcomes) / np.sqrt(num_dialogs))
    
#END OF FILE

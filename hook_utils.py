# -*- coding: utf-8 -*-
import tensorflow as tf
import os

class CheckpointSaverHook(tf.train.SessionRunHook):
    def __init__(self, checkpoint_path_dict, var_scopes_dict=None,  
                checkpoint_exclude_scopes_dict=None):
        """Generate a CheckpointSaverHook to restore checkpoints for the models.
        
        Args:
            checkpoint_path_dict: A dictionary of checkpoint paths.
            var_scopes_dict: A dictionary of var_scopes we want to read .
            checkpoint_exclude_scopes_dict: A dictionary of comma-separated list of 
                scopes of variables to exclude when restoring from a checkpoint.
        Returns:
            A hooks for training models with estimator.
        """
        
        tf.logging.info("Create RestoreCheckpointHook.")
        #super(IteratorInitializerHook, self).__init__()
        self.checkpoint_path_dict = checkpoint_path_dict
        
        self.var_scopes_dict=var_scopes_dict
        self.checkpoint_exclude_scopes_dict=checkpoint_exclude_scopes_dict

        print(self.checkpoint_path_dict, self.var_scopes_dict, self.checkpoint_exclude_scopes_dict)
    
        
    def begin(self):
        # You can add ops to the graph here.
        print('Before starting the session.')
        
        # 1. Create saver
        #exclusions = []
        # if self.checkpoint_exclude_scopes:
        #  exclusions = [scope.strip()
        #                for scope in self.checkpoint_exclude_scopes.split(',')]
        #
        #variables_to_restore = []
        # for var in slim.get_model_variables(): #tf.global_variables():
        #  excluded = False
        #  for exclusion in exclusions:
        #    if var.op.name.startswith(exclusion):
        #      excluded = True
        #      break
        #  if not excluded:
        #    variables_to_restore.append(var)
        # inclusions
        #[var for var in tf.trainable_variables() if var.op.name.startswith('InceptionResnetV1')]
        
        self.init_savers = retrieve_init_savers(self.var_scopes_dict,self.checkpoint_exclude_scopes_dict )

#         variables_to_restore = tf.contrib.framework.filter_variables(
#             slim.get_model_variables(),
#             include_patterns=self.include_scope_patterns,  # ['Conv'],
#             # ['biases', 'Logits'],
#             exclude_patterns=self.exclude_scope_patterns,

#             # If True (default), performs re.search to find matches
#             # (i.e. pattern can match any substring of the variable name).
#             # If False, performs re.match (i.e. regexp should match from the beginning of the variable name).
#             reg_search=True
#         )
#         self.saver = tf.train.Saver(variables_to_restore)

    def after_create_session(self, session, coord):
        # When this is called, the graph is finalized and
        # ops can no longer be added to the graph.
        print('Session created.')
        print(self.checkpoint_path_dict)
        for key,path in self.checkpoint_path_dict.items():
            tf.logging.info('Fine-tuning from %s' % path)
            current_saver = self.init_savers[key]
            current_saver.restore(session, os.path.expanduser(path))
            tf.logging.info('End fineturn from %s' % path)

    def before_run(self, run_context):
        #print('Before calling session.run().')
        return None  # SessionRunArgs(self.your_tensor)

    def after_run(self, run_context, run_values):
        #print('Done running one step. The value of my tensor: %s', run_values.results)
        # if you-need-to-stop-loop:
        #  run_context.request_stop()
        pass

    def end(self, session):
        #print('Done with the session.')
        pass
    
    
def retrieve_init_savers(var_scopes_dict=None, 
                         checkpoint_exclude_scopes_dict=None):
    """Retrieve a dictionary of all the initial savers for the models.
    
    Args:
        var_scopes_dict: A dictionary of variable scopes for the models.
        checkpoint_exclude_scopes_dict: A dictionary of comma-separated list of 
            scopes of variables to exclude when restoring from a checkpoint.
        
    Returns:
        A dictionary of init savers.
    """
    if var_scopes_dict is None:
        return None
    # Dictionary of init savers
    init_savers = {}
    for key, scope in var_scopes_dict.items():
        trainable_vars = [
            v for v in tf.trainable_variables() if v.op.name.startswith(scope)]
        
        exclusions = []
        if not checkpoint_exclude_scopes_dict:
            checkpoint_exclude_scopes = None
        else:
            checkpoint_exclude_scopes = checkpoint_exclude_scopes_dict.get(
            key, None)
        if checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in 
                         checkpoint_exclude_scopes.split(',')]
            print("exclusions", exclusions)
        variables_to_restore = []
        for var in trainable_vars:
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    print("var.op.name", var.op.name)
            if not excluded:
                variables_to_restore.append(var)
        
        init_saver = tf.train.Saver(var_list=variables_to_restore)
        init_savers[key] = init_saver
    return init_savers

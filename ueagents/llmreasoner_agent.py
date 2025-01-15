class LLMReasonerAgent(object):
    """Agent using LLM Reasoner"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

    def reset(self, agent_json, world_json):
        return
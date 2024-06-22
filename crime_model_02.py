from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

class PotentialCriminal(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = model.wealth_arr[unique_id]
        self.criminal = False
        self.wealth = model.wealth_arr[unique_id]
        
    def step(self):
        if self.criminal:  
            return

        other_agent = self.random.choice(self.model.schedule.agents)

        if other_agent.wealth * self.model.deterrence - self.wealth > 0: # using relative deterrence
            self.criminal = True

            stolen_amount = self.random.random() * other_agent.wealth * self.model.deterrence # using relative amount
            self.wealth += stolen_amount 
            other_agent.wealth -= stolen_amount
        
        return

def get_crime_rate(model):
    agent_out = [agent.criminal for agent in model.schedule.agents]
    return sum(agent_out)/model.num_agents

def get_gini(model):
    from distribution import gini
    return gini(model.wealth_arr, plot=False)
    
class CrimeModel(Model):
    def __init__(self, N, deterrence, wealth_arr, fraction_stolen): # added parameter for fraction_stolen
        self.num_agents = N
        self.deterrence = deterrence
        self.wealth_arr = wealth_arr
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"Crime rate": get_crime_rate,
                             "Gini Coefficient": get_gini    # added gini collector
                             },
            agent_reporters={"Criminal": "criminal", "Wealth": "wealth"}
        )
        
        for i in range(self.num_agents):
            a = PotentialCriminal(i, self)
            self.schedule.add(a)
    
    def crime_rate(self):
        return sum([a.criminal for a in self.schedule.agents])/self.num_agents
            
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
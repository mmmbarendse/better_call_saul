[Project Requirements](https://github.com/pjercic/ComputationalModellingSocialSystems2024/blob/main/ProjectGuide/ProjectsGuide.md)

[Document](https://www.overleaf.com/project/665e05ec6795d650709e27dd)

### Report Outline

- **Motivation:** research question including some references to argue about previous work
- **Model description:** precise explanation of all dynamics and assumptions and what other models it is based on. If any data is involved, please explain it here.
- **Model analysis:** Visualization of the model dynamics to provide insights and results of simulations
- **Conclusions:** reflect on your results and how they inform the question that motivated the project

### Model 1
Model Parameters: Deterrence (Punishment)
Agent Parameter: 
- Wealth (drawn from Gini) +- some (weighted) nosie (disposition towards crime)

Crime is binary (0; 1)

**Step Function**: 
- Draw another agent
- Compare Wealth A_1 - Wealth A_2; if Difference > Deterrence + Noise of Agent 1; -> Crime  

-> in one Step Agent can only become criminal once

### Model 2
I0: all Agents are given some Wealth from Gini
I0: all Agents are given some Income correlated to Wealth (+ Noise)
I0: all Agents are given some Expenses correlated to Wealth (+ Noise)

I_k: Wealth_k = Wealth_k-1 + Income_k-1 - Expenses_k-1

2 mechanisms: 
- if wealth < 0: agent will steal
- if difference > deterrence + disposition -> theft

### Model 3
**Ideas:** 
- Same initialization as above
- Agent has living standard which depends on neighbourhood
- Breaking bad depends on number of iterations where living standard cannot be fulfilled

Potential Parameters: Unemployment, GDP, Welfare 

---
layout: post
title: "Multi-Robot Constraints"
katex: true
tags: [Research, RL, Human-Robot Interactions, Multi-Robot]
---
## Human and Multi-Robot Interactions 
Currently, there isn't a lot of exploration in how humans and agents interact on a multi-agent level. It remains a significant challenge to integrate human operators in multi-agent systems. I want to investigate how humans interact with a multi-agent system through constraints.

Another possible motivation: constraints given by humans are often not detailed enough, so most constraints end up being rather ambiguous. What could be contributed here is a framework that grounds these constraints in an MMDP and actively clarifies interpretations based on the consequences of the constraint. Then we can create hard constraints through shielding and soft constraints through reward shaping.

## Background
While MARL and HRI have advanced independently, their effective integration, particularly regarding human influence on MARL agent decision-making has not been heavily explored. The question to answer is how do humans give constraints to multi-agent systems and would that help them learn quicker or better? Additionally, feedback especially given as constraints is likely difficult to give as most people would probably formulate their feedback as advice to the agents. There is recent work [2] that explored constraints but they remain in single agent.

Vagueness or ambiguity in the feedback also poses a significant challenge as it makes it difficult to label constraint types.

## User Study
I will conduct a user study online by showing users a four room environment where there are small items placed in each of the four rooms, but one room has a larger item that requires two agents to pick up and deliver. The goal is to deliver the big item in addition to the smaller items. We ask users to give constraints that will help the robots avoid behavior that causes problems for the agents and help them complete the task better.

The users give their feedback to an LLM agent (running qwen3.6:35b) that is engineered to label their feedback as hard, soft, or ambiguous/not relevant. If a user gives constraints that are vague or ambiguous, they are asked to clarify, and the agent also explains what clarification it needs. If the user enters a piece of advice and not a constraint, they are asked to re-frame it as a constraint. All information sent to the agent is labeled and saved. This will be used to help train the LLM labeling assistant to label better.
---

## References

[1] Boutilier, C. Planning, Learning and Coordination in Multiagent Decision Processes.

[2] Kuehn, H., Santos, L., & Leite, I. (2026). Clarifying Constraints in Interactive Robot Learning with Language Feedback. *Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction*. Edinburgh, Scotland, UK: ACM, pp. 816–824. doi: [10.1145/3757279.3785583](https://dl.acm.org/doi/10.1145/3757279.3785583)

[3] Melcer, D. (2025). Safe Multi-Agent Learning via Shielding in Decentralized Environments.

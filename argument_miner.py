from copy import deepcopy

import Uncertainpy.src.uncertainpy.gradual as grad


class ArgumentMiner:
    def __init__(
        self, llm_manager, generate_prompt, depth=1, breadth=1, generation_args={}
    ):
        self.depth = depth
        self.breadth = breadth
        self.llm_manager = llm_manager
        self.generate_prompt = generate_prompt
        self.generation_args = generation_args

    def generate_args_for_parent(self, parent, name, base_score_generator):
        """
        Generates up to `breadth` supporting and `breadth` attacking arguments for the parent
        in a single LLM call and adds them to the argument tree. Slots the LLM declined (N/A)
        are not added, as they would carry a base score of 0 and thus no influence anyway.
        """
        prompt, constraints, format_args = self.generate_prompt(
            parent.get_arg(), breadth=self.breadth
        )
        # The token budget in generation_args is per argument; the joint call produces
        # 2 * breadth arguments (plus their labels) in one completion
        generation_args = dict(self.generation_args)
        generation_args["max_new_tokens"] = (
            generation_args.get("max_new_tokens", 128) * 2 * self.breadth
        )
        supports, attacks = format_args(
            self.llm_manager.chat_completion(
                prompt,
                print_result=True,
                trim_response=True,
                **constraints,
                **generation_args,
            ),
            prompt,
        )

        children = []
        for prefix, args, support in (("S", supports, True), ("A", attacks, False)):
            for b, arg in enumerate(args, start=1):
                base_score = base_score_generator(
                    arg, claim=parent.get_arg(), support=support
                )
                child = grad.Argument(f"{prefix}{name}b{b}", arg, float(base_score))
                if support:
                    self.argument_tree.add_support(child, parent)
                else:
                    self.argument_tree.add_attack(child, parent)
                children.append(child)
        return children

    def generate_arguments(self, statement, base_score_generator):
        """Generates arguments for and against a statement, up to the given breadth and depth."""
        self.argument_tree = grad.BAG()
        topic = grad.Argument(f"db0", statement, 0.5)
        # Register the topic explicitly so it exists even if no arguments are generated
        self.argument_tree.arguments[topic.name] = topic
        topic_base_score = base_score_generator(statement, topic=True)

        previous_layer = [topic]
        for d in range(1, self.depth + 1):
            new_layer = []
            for p in previous_layer:
                new_layer += self.generate_args_for_parent(
                    parent=p,
                    name=f"{p.name}←d{d}",
                    base_score_generator=base_score_generator,
                )
            previous_layer = new_layer

        topic_base_score_bag = deepcopy(self.argument_tree)
        topic_base_score_bag.arguments[topic.name].reset_initial_weight(
            topic_base_score
        )

        return self.argument_tree, topic_base_score_bag

    """If argument is similar to other arguments in same branch then we cut of that argument."""

    def cut_arguments(self, arguments):
        pass

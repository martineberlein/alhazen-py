from sklearn.tree import DecisionTreeClassifier

from isla.parser import EarleyParser

from alhazen.features import Feature, NumericInterpretation, LengthFeature, ExistenceFeature
from alhazen.input_specifications import SPECIFICATION_GRAMMAR, create_new_input_specification, InputSpecification, \
    Requirement
from alhazen.requirementExtractionDT.requirements import tree_to_paths, TreePath


def _requirement_to_isla_string(requirement: Requirement) -> str:
    """
            We use the extended syntax of ISLA
            - 1D:
                    - exists(<digit>)                       ???
                    - num(<number>) </>/<=/>= xyz           str.to.int(<number>) < 12
                    - len(<function>) </>/<=/>= xyz         str.len(<function>) > 3.5
            - 2D:
                1. f.key is terminal:
                    - exists(<function> == sqrt)            <function> = "sqrt"
                    - exists(<maybe_minus>) == )            <maybe_minus> = ""
                2. f.key is non-terminal:
                    - exists(<function> == .<digit>)        ???
            """
    feature: Feature = requirement.feature
    constraint: str = ""
    if feature.rule == feature.key:
        # 1D Case
        if isinstance(feature, NumericInterpretation):
            constraint = f"str.to.int({feature.rule}) {requirement.quant} {requirement.value}"
        if isinstance(feature, LengthFeature):
            constraint = f"str.len({feature.rule}) {requirement.quant} {requirement.value}"
    else:
        if isinstance(feature, ExistenceFeature):
            constraint = f'''{feature.rule} = "{feature.key}"'''

    return constraint


def input_specification_to_isla_constraint(input_specification: InputSpecification) -> str:
    constraints = []
    for idx, requirement in enumerate(input_specification.requirements):
        constraint = _requirement_to_isla_string(requirement)
        constraints.append(constraint)

    isla_string = " and ".join(constraints)

    return isla_string


def decision_tree_to_isla_constraint(clf: DecisionTreeClassifier, all_features, bug_triggering: bool = True) -> str:
    feature_names = [f.name for f in all_features]
    all_paths = tree_to_paths(clf, feature_names)

    relevant_paths: set[TreePath] = set()

    for path in all_paths:
        if path.is_bug() == bug_triggering:
            relevant_paths.add(path)

    parser = EarleyParser(SPECIFICATION_GRAMMAR)
    input_specification_list = []

    for path in relevant_paths:
        try:
            for tree in parser.parse(path.get_requirements_as_string()):
                input_specification_list.append(
                    create_new_input_specification(tree, all_features)
                )
        except SyntaxError as e:
            # Catch Parsing Syntax Errors: num(<term>) in [-900, 0] will fail; Might fix later
            # For now, inputs following that form will be ignored
            print(e)
            pass

    constraints = []
    for input_specification in input_specification_list:
        constraints.append(input_specification_to_isla_constraint(input_specification))

    isla_string = " and ".join(constraints)

    return isla_string

from app import initial_state, graph

CASES = [
    {
        "question": "How does Project Atlas prevent validation data from contaminating preprocessing?",
        "expect_abstain": False,
    },
    {
        "question": "When should Project Atlas training stop?",
        "expect_abstain": False,
    },
    {
        "question": "Which AWS region hosts Project Atlas?",
        "expect_abstain": True,
    },
]


def main() -> None:
    passed = 0
    for case in CASES:
        result = graph.invoke(initial_state(case["question"]))
        abstained = result["answer"].startswith(
            "I could not find enough relevant evidence"
        )
        case_passed = abstained == case["expect_abstain"]
        passed += int(case_passed)

        status = "PASS" if case_passed else "FAIL"
        print(f"{status}: {case['question']}")
        if not case_passed:
            for event in result["trace"]:
                print(f"  - {event}")

    print(f"\nScore: {passed}/{len(CASES)}")


if __name__ == "__main__":
    main()
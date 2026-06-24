from src.agent import ask, build_agent

CASES = [
    ("Were Scott Derrickson and Ed Wood of the same nationality?", "Yes (American)"),
    ('The director of the romantic comedy "Big Stone Gap" is based in what New York city?',
     "Greenwich Village, New York City"),
    ("What government position was held by the woman who portrayed Corliss Archer in the film "
     "Kiss and Tell?", "Chief of Protocol"),
]


def main():
    agent = build_agent()
    for question, gold in CASES:
        r = ask(agent, question)
        print("=" * 70)
        print("Q:", question)
        print("GOLD:", gold)
        print(f"route={r['route']}  tries={r['tries']}  final grade={r['grade']}")
        if r["question"] != question:
            print("rewritten query:", r["question"])
        print("ANSWER:", r["answer"].strip())


if __name__ == "__main__":
    main()

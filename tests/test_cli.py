from ragagent.cli import main


def test_cli_ask(capsys):
    assert main(["ask", "What is the return window?"]) == 0
    out = capsys.readouterr().out
    assert "status: answered" in out
    assert "30" in out


def test_cli_ask_abstains(capsys):
    assert main(["ask", "What is the quarterly revenue of Northwind "
                        "Outfitters?"]) == 0
    assert "status: abstained" in capsys.readouterr().out


def test_cli_eval(capsys):
    assert main(["eval"]) == 0
    out = capsys.readouterr().out
    assert "answer accuracy" in out
    assert "recall@k" in out


def test_cli_policies(capsys):
    assert main(["policies"]) == 0
    assert "v1.0.0" in capsys.readouterr().out


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    assert "scenarios passed" in capsys.readouterr().out

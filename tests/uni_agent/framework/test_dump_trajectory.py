import json

import numpy as np

from examples.tools.dump_trajectory import main


def test_dump_trajectory_writes_readable_summary_and_token_table(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "step_7" / "session-sample-0-rollout-0-test"
    run_dir.mkdir(parents=True)
    np.savez_compressed(
        run_dir / "trajectory.npz",
        traj0_prompt_ids=np.asarray([10, 11], dtype=np.int32),
        traj0_response_ids=np.asarray([20, 21, 22], dtype=np.int32),
        traj0_response_mask=np.asarray([1, 0, 1], dtype=np.int8),
        traj0_response_logprobs=np.asarray([-0.1, 0.0, -0.2], dtype=np.float32),
    )
    (run_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "session_id": "session-sample-0-rollout-0-test",
                "num_trajectories": 1,
                "trajectories": [
                    {
                        "num_turns": 2,
                        "prompt_len": 2,
                        "response_len": 3,
                        "model_token_count": 2,
                        "reward_score": 0.5,
                        "reward_info": {"resolved": True},
                        "materialization_reason": "finalize",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "decoded"
    monkeypatch.setattr(
        "sys.argv",
        ["dump_trajectory.py", str(run_dir), "--out", str(output_dir), "--max-turns", "2"],
    )

    assert main() == 0

    stdout = capsys.readouterr().out
    assert "reward=0.5" in stdout
    assert "SEGMENT-1 [MODEL]" in stdout
    assert "SEGMENT-2 [CONTEXT]" in stdout
    assert "1 response segments omitted" in stdout
    assert (output_dir / "trajectory.txt").is_file()
    token_table = (output_dir / "tokens.tsv").read_text(encoding="utf-8")
    assert "MODEL-1\t20" in token_table
    assert "CONTEXT\t21" in token_table
    assert token_table.rstrip().endswith("1\t0.5")

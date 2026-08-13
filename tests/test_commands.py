import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zordon.agent import Agent
from zordon.audit import AuditLog
from zordon.commands import CommandContext, CommandRouter
from zordon.config import Config
from zordon.memory import MemoryStore


class NoopProvider:
    def stream_step(self, system_prompt, input_text, tools):
        yield "Noop."
        return []


class CommandsTest(unittest.TestCase):
    def test_router_handles_memory_without_cli_loop(self):
        router = _router()

        remembered = router.handle("/remember tone cool and calm")
        listed = router.handle("/memory")

        self.assertTrue(remembered.handled)
        self.assertIn("Remembered tone", remembered.output)
        self.assertIn("tone: cool and calm", listed.output)

    def test_router_stages_and_cancels_draft_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drafts = root / "drafts"
            drafts.mkdir()
            path = drafts / "idea.md"
            path.write_text("Opening hook.", encoding="utf-8")
            router = _router(root=root)

            staged = router.handle("/delete-draft idea.md")
            cancelled = router.handle("/cancel")

            self.assertIn("Delete draft idea.md?", staged.output)
            self.assertEqual(cancelled.output, "Cancelled delete-draft idea.md.")
            self.assertTrue(path.exists())

    def test_router_prints_latest_draft(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drafts = root / "drafts"
            drafts.mkdir()
            older = drafts / "older.md"
            newer = drafts / "newer.md"
            older.write_text("Old draft.", encoding="utf-8")
            newer.write_text("New draft.", encoding="utf-8")
            router = _router(root=root)

            outcome = router.handle("/latest drafts")

        self.assertTrue(outcome.handled)
        self.assertIn("newer.md", outcome.output)
        self.assertIn("New draft.", outcome.output)

    def test_router_lists_and_shows_latest_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            image = images / "cover.png"
            image.write_bytes(b"png")
            router = _router(root=root)

            listed = router.handle("/images")
            latest = router.handle("/latest images")

        self.assertTrue(listed.handled)
        self.assertIn("cover.png", listed.output)
        self.assertTrue(latest.handled)
        self.assertIn("cover.png", latest.output)

    def test_router_stages_and_opens_latest_image_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            image = images / "cover.png"
            image.write_bytes(b"png")
            router = _router(root=root)

            staged = router.handle("/open latest image")
            with patch("zordon.commands.open_image") as open_image:
                confirmed = router.handle("confirm")

        self.assertTrue(staged.handled)
        self.assertIn("Open latest image cover.png?", staged.output)
        open_image.assert_called_once_with(image)
        self.assertIn("Opened image", confirmed.output)

    def test_router_can_cancel_open_latest_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            image = images / "cover.png"
            image.write_bytes(b"png")
            router = _router(root=root)

            router.handle("/open latest image")
            cancelled = router.handle("cancel")

        self.assertEqual(cancelled.output, "Cancelled open-image cover.png.")

    def test_router_exits_on_exit_command(self):
        outcome = _router().handle("exit")

        self.assertTrue(outcome.handled)
        self.assertTrue(outcome.should_exit)
        self.assertEqual(outcome.output, "Goodbye.")

    def test_router_accepts_plain_cancel_for_pending_delete(self):
        router = _router()
        router.pending_delete_draft = "idea.md"

        outcome = router.handle("cancel")

        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.output, "Cancelled delete-draft idea.md.")

    def test_router_accepts_common_confirm_typo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drafts = root / "drafts"
            drafts.mkdir()
            path = drafts / "idea.md"
            path.write_text("Opening hook.", encoding="utf-8")
            router = _router(root=root)
            router.pending_delete_draft = "idea.md"

            outcome = router.handle("confrim")

        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.output, "Deleted draft idea.md.")
        self.assertFalse(path.exists())

    def test_router_kill_switch_exits_and_records_audit_event(self):
        audit = AuditLog(events=[])
        router = _router(audit=audit)
        router.pending_delete_draft = "idea.md"

        outcome = router.handle("/Kill")

        self.assertTrue(outcome.handled)
        self.assertTrue(outcome.should_exit)
        self.assertIn("Kill switch engaged", outcome.output)
        self.assertIsNone(router.pending_delete_draft)
        self.assertIn("kill_switch", [event["event"] for event in audit.events])

    def test_router_reports_version(self):
        outcome = _router().handle("/version")

        self.assertTrue(outcome.handled)
        self.assertIn("Zordon 0.6.0", outcome.output)
        self.assertIn("Tier 6 local assistant harness", outcome.output)
        self.assertIn("python:", outcome.output)

    def test_router_reports_short_commands(self):
        outcome = _router().handle("/commands")

        self.assertTrue(outcome.handled)
        self.assertIn("Zordon commands", outcome.output)
        self.assertIn("/ready", outcome.output)
        self.assertIn("/recent", outcome.output)
        self.assertIn("/open-images", outcome.output)
        self.assertIn("/content-plan", outcome.output)
        self.assertIn("/project-next", outcome.output)
        self.assertIn("/project-brief-next", outcome.output)

    def test_router_runs_doctor(self):
        outcome = _router().handle("/doctor")

        self.assertTrue(outcome.handled)
        self.assertIn("Zordon doctor:", outcome.output)
        self.assertIn("config", outcome.output)
        self.assertIn("tools", outcome.output)

    def test_router_reports_state_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outcome = _router(root=root).handle("/state")

        self.assertTrue(outcome.handled)
        self.assertIn("Zordon state", outcome.output)
        self.assertIn("state root:", outcome.output)
        self.assertIn("memory.jsonl", outcome.output)

    def test_router_reports_today_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drafts = root / "drafts"
            images = root / "images"
            logs = root / "logs"
            drafts.mkdir()
            images.mkdir()
            logs.mkdir()
            (drafts / "today.md").write_text("Draft.", encoding="utf-8")
            (images / "today.png").write_bytes(b"png")
            log_path = logs / "zordon-today.jsonl"
            log_path.write_text(
                '{"event":"user_turn"}\n{"event":"assistant_reply"}\n', encoding="utf-8"
            )
            audit = AuditLog(path=log_path)
            memory = MemoryStore(entries=[])
            memory.remember("favorite_color", "green", "manual")
            router = _router(root=root, audit=audit, memory=memory)

            outcome = router.handle("/today")

        self.assertTrue(outcome.handled)
        self.assertIn("Zordon today", outcome.output)
        self.assertIn("events today: 2", outcome.output)
        self.assertIn("memories: 1", outcome.output)
        self.assertIn("latest draft: today.md", outcome.output)
        self.assertIn("latest image: today.png", outcome.output)

    def test_router_reports_ready_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drafts = root / "drafts"
            images = root / "images"
            drafts.mkdir()
            images.mkdir()
            (drafts / "ready.md").write_text("Draft.", encoding="utf-8")
            (images / "ready.png").write_bytes(b"png")
            memory = MemoryStore(entries=[])
            memory.remember("favorite_color", "green", "manual")
            router = _router(root=root, memory=memory)

            outcome = router.handle("/ready")

        self.assertTrue(outcome.handled)
        self.assertIn("Zordon ready", outcome.output)
        self.assertIn("memory: 1 entries", outcome.output)
        self.assertIn("latest draft: ready.md", outcome.output)
        self.assertIn("latest image: ready.png", outcome.output)
        self.assertIn("tools:", outcome.output)

    def test_router_creates_lists_uses_and_opens_project(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)

            created = router.handle("/new-project Green Campaign")
            listed = router.handle("/projects")
            current = router.handle("/project")
            switched = router.handle("/use-project green-campaign")
            switched_without_the = router.handle("/use-project the green-campaign")
            staged = router.handle("/open-project")
            with patch(
                "zordon.commands.open_path", return_value=root / "projects" / "green-campaign"
            ) as open_path:
                confirmed = router.handle("confirm")

        self.assertIn("Created project Green Campaign (green-campaign).", created.output)
        self.assertIn("green-campaign: Green Campaign *", listed.output)
        self.assertIn("name: Green Campaign", current.output)
        self.assertIn("Using project Green Campaign (green-campaign).", switched.output)
        self.assertIn("Using project Green Campaign (green-campaign).", switched_without_the.output)
        self.assertEqual(staged.output, "Open project folder? Say Confirm Or Deny.")
        open_path.assert_called_once_with(
            root / "projects" / "green-campaign", create_directory=True
        )
        self.assertIn("Opened project folder", confirmed.output)

    def test_router_copies_and_opens_project_drafts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drafts = root / "drafts"
            drafts.mkdir()
            draft = drafts / "global.md"
            draft.write_text("Global draft.", encoding="utf-8")
            router = _router(root=root)
            router.handle("/new-project Green Campaign")

            saved = router.handle("/save-latest-draft-to-project")
            listed = router.handle("/project-drafts")
            staged_folder = router.handle("/open-project-drafts")
            with patch(
                "zordon.commands.open_path",
                return_value=root / "projects" / "green-campaign" / "drafts",
            ) as open_path:
                confirmed_folder = router.handle("confirm")
            staged_file = router.handle("/open-latest-project-draft")
            with patch(
                "zordon.commands.open_path",
                return_value=root / "projects" / "green-campaign" / "drafts" / "global.md",
            ) as open_path_file:
                confirmed_file = router.handle("confirm")

        self.assertIn("Saved latest draft to project: global.md.", saved.output)
        self.assertIn("global.md", listed.output)
        self.assertEqual(staged_folder.output, "Open project drafts folder? Say Confirm Or Deny.")
        open_path.assert_called_once_with(
            root / "projects" / "green-campaign" / "drafts", create_directory=True
        )
        self.assertIn("Opened project drafts folder", confirmed_folder.output)
        self.assertEqual(
            staged_file.output, "Open latest project draft global.md? Say Confirm Or Deny."
        )
        open_path_file.assert_called_once_with(
            root / "projects" / "green-campaign" / "drafts" / "global.md", create_directory=False
        )
        self.assertIn("Opened latest project draft", confirmed_file.output)

    def test_router_copies_and_opens_project_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            image = images / "cover.png"
            image.write_bytes(b"png")
            router = _router(root=root)
            router.handle("/new-project Green Campaign")

            saved = router.handle("/save-latest-image-to-project")
            listed = router.handle("/project-images")
            staged_folder = router.handle("/open-project-images")
            with patch(
                "zordon.commands.open_path",
                return_value=root / "projects" / "green-campaign" / "images",
            ) as open_path:
                confirmed_folder = router.handle("confirm")
            staged_file = router.handle("/open-latest-project-image")
            with patch(
                "zordon.commands.open_path",
                return_value=root / "projects" / "green-campaign" / "images" / "cover.png",
            ) as open_path_file:
                confirmed_file = router.handle("confirm")

        self.assertIn("Saved latest image to project: cover.png.", saved.output)
        self.assertIn("cover.png", listed.output)
        self.assertEqual(staged_folder.output, "Open project images folder? Say Confirm Or Deny.")
        open_path.assert_called_once_with(
            root / "projects" / "green-campaign" / "images", create_directory=True
        )
        self.assertIn("Opened project images folder", confirmed_folder.output)
        self.assertEqual(
            staged_file.output, "Open latest project image cover.png? Say Confirm Or Deny."
        )
        open_path_file.assert_called_once_with(
            root / "projects" / "green-campaign" / "images" / "cover.png", create_directory=False
        )
        self.assertIn("Opened latest project image", confirmed_file.output)

    def test_router_sets_shows_and_opens_project_brief(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")

            empty = router.handle("/project-brief")
            saved = router.handle("/set-project-brief A calm green campaign for launch content.")
            shown = router.handle("/project brief")
            staged = router.handle("/open-project-brief")
            with patch(
                "zordon.commands.open_path",
                return_value=root / "projects" / "green-campaign" / "brief.md",
            ) as open_path:
                confirmed = router.handle("confirm")

        self.assertIn("No project brief saved yet.", empty.output)
        self.assertEqual(saved.output, "Saved project brief to brief.md.")
        self.assertIn("Project brief: Green Campaign", shown.output)
        self.assertIn("A calm green campaign", shown.output)
        self.assertEqual(staged.output, "Open project brief? Say Confirm Or Deny.")
        open_path.assert_called_once_with(
            root / "projects" / "green-campaign" / "brief.md", create_directory=False
        )
        self.assertIn("Opened project brief", confirmed.output)

    def test_router_reports_project_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")
            router.handle("/set-project-brief Calm green launch.")
            project_root = root / "projects" / "green-campaign"
            (project_root / "drafts" / "draft.md").write_text("Draft.", encoding="utf-8")
            (project_root / "images" / "cover.png").write_bytes(b"png")

            outcome = router.handle("/project-status")

        self.assertTrue(outcome.handled)
        self.assertIn("Project status", outcome.output)
        self.assertIn("name: Green Campaign", outcome.output)
        self.assertIn("brief: saved", outcome.output)
        self.assertIn("drafts: 1", outcome.output)
        self.assertIn("latest draft: draft.md", outcome.output)
        self.assertIn("images: 1", outcome.output)
        self.assertIn("latest image: cover.png", outcome.output)
        self.assertIn(str(project_root), outcome.output)

    def test_router_builds_saves_and_opens_content_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")
            router.handle("/set-project-brief Calm green launch content for Zordon.")
            project_root = root / "projects" / "green-campaign"
            (project_root / "drafts" / "intro.md").write_text("Intro draft.", encoding="utf-8")
            (project_root / "images" / "cover.png").write_bytes(b"png")

            plan = router.handle("/content-plan")
            staged_save = router.handle("/save-content-plan")
            confirmed_save = router.handle("confirm")
            saved_path = project_root / "notes" / "content-plan.md"
            saved_exists = saved_path.exists()
            saved_content = saved_path.read_text(encoding="utf-8")
            staged_open = router.handle("/open-content-plan")
            with patch("zordon.commands.open_path", return_value=saved_path) as open_path:
                confirmed_open = router.handle("confirm")

        self.assertTrue(plan.handled)
        self.assertIn("Content plan: Green Campaign", plan.output)
        self.assertIn("Project goal", plan.output)
        self.assertIn("Content ideas", plan.output)
        self.assertIn("Image concepts", plan.output)
        self.assertIn("Caption angles", plan.output)
        self.assertIn("Next draft", plan.output)
        self.assertIn("Calm green launch content", plan.output)
        self.assertIn("latest draft: intro.md", plan.output.lower())
        self.assertIn("latest image: cover.png", plan.output.lower())
        self.assertEqual(
            staged_save.output, "Save content plan to content-plan.md? Say Confirm Or Deny."
        )
        self.assertTrue(saved_exists)
        self.assertIn("Content plan: Green Campaign", saved_content)
        self.assertIn(f"Saved content plan to {saved_path}.", confirmed_save.output)
        self.assertEqual(staged_open.output, "Open content plan? Say Confirm Or Deny.")
        open_path.assert_called_once_with(saved_path, create_directory=False)
        self.assertIn("Opened content plan", confirmed_open.output)

    def test_router_reports_missing_content_plan_before_open(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")

            outcome = router.handle("/open-content-plan")

        self.assertEqual(outcome.output, "No content plan saved yet.")

    def test_router_reports_no_active_project_for_content_plan(self):
        outcome = _router().handle("/content-plan")

        self.assertEqual(outcome.output, "No active project.")

    def test_router_project_brief_next_returns_draft_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")
            router.handle("/set-project-brief Calm green launch content for Zordon.")

            outcome = router.handle("/project-brief-next")

        self.assertTrue(outcome.handled)
        self.assertIn("Project brief next", outcome.output)
        self.assertIn("project: Green Campaign", outcome.output)
        self.assertIn("prompt:", outcome.output)
        self.assertIn("Draft a short content piece for Green Campaign.", outcome.output)
        self.assertIn("Calm green launch content for Zordon.", outcome.output)
        self.assertIn("strong opening line", outcome.output)

    def test_router_project_brief_next_requires_brief(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")

            outcome = router.handle("/project brief next")

        self.assertEqual(
            outcome.output, "No project brief saved yet. Use /set-project-brief <text> first."
        )

    def test_router_project_next_requires_brief_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")

            outcome = router.handle("/project-next")

        self.assertIn("Project next", outcome.output)
        self.assertIn("action: Set a short project brief", outcome.output)
        self.assertIn("/set-project-brief <one sentence>", outcome.output)

    def test_router_project_next_requires_content_plan_after_brief(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")
            router.handle("/set-project-brief Calm green launch.")

            outcome = router.handle("/project next")

        self.assertIn("Project next", outcome.output)
        self.assertIn("action: Create and save a content plan", outcome.output)
        self.assertIn("/content-plan, then /save-content-plan", outcome.output)

    def test_router_project_next_recommends_packaging_when_assets_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)
            router.handle("/new-project Green Campaign")
            router.handle("/set-project-brief Calm green launch.")
            router.handle("/content-plan")
            router.handle("/save-content-plan")
            router.handle("confirm")
            project_root = root / "projects" / "green-campaign"
            (project_root / "drafts" / "intro.md").write_text("Intro draft.", encoding="utf-8")
            (project_root / "images" / "cover.png").write_bytes(b"png")

            outcome = router.handle("/project-next")

        self.assertIn("Project next", outcome.output)
        self.assertIn("project: Green Campaign", outcome.output)
        self.assertIn(
            "action: Turn intro.md and cover.png into one publishable post.", outcome.output
        )
        self.assertIn("command: Say: Refine this into a social post.", outcome.output)

    def test_router_reports_recent_turns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            log_path = logs / "zordon-today.jsonl"
            log_path.write_text(
                '{"event":"user_turn","text":"Make a draft"}\n'
                '{"event":"assistant_reply","text":"Draft ready"}\n'
                '{"event":"voice_transcript","text":"Slash today"}\n',
                encoding="utf-8",
            )
            router = _router(root=root, audit=AuditLog(path=log_path))

            outcome = router.handle("/recent")

        self.assertTrue(outcome.handled)
        self.assertIn("Zordon recent", outcome.output)
        self.assertIn("You: Make a draft", outcome.output)
        self.assertIn("Zordon: Draft ready", outcome.output)
        self.assertIn("You: Slash today", outcome.output)

    def test_router_stages_and_opens_state_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)

            staged = router.handle("/open-state")
            with patch("zordon.commands.open_state_folder", return_value=root) as open_state_folder:
                confirmed = router.handle("confirm")

        self.assertTrue(staged.handled)
        self.assertEqual(staged.output, "Open Zordon state folder? Say Confirm Or Deny.")
        open_state_folder.assert_called_once_with(root)
        self.assertIn("Opened state folder", confirmed.output)

    def test_router_can_cancel_open_state(self):
        router = _router()

        staged = router.handle("/open state")
        cancelled = router.handle("deny")

        self.assertTrue(staged.handled)
        self.assertEqual(cancelled.output, "Cancelled open-state.")

    def test_router_stages_and_opens_drafts_folder_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)

            staged = router.handle("/open-drafts")
            with patch("zordon.commands.open_path", return_value=root / "drafts") as open_path:
                confirmed = router.handle("confirm")

        self.assertEqual(staged.output, "Open drafts folder? Say Confirm Or Deny.")
        open_path.assert_called_once_with(root / "drafts", create_directory=True)
        self.assertIn("Opened drafts folder", confirmed.output)

    def test_router_stages_and_opens_images_folder_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)

            staged = router.handle("/open images")
            with patch("zordon.commands.open_path", return_value=root / "images") as open_path:
                confirmed = router.handle("confirm")

        self.assertEqual(staged.output, "Open images folder? Say Confirm Or Deny.")
        open_path.assert_called_once_with(root / "images", create_directory=True)
        self.assertIn("Opened images folder", confirmed.output)

    def test_router_stages_and_opens_logs_folder_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = _router(root=root)

            staged = router.handle("/open-logs")
            with patch("zordon.commands.open_path", return_value=root / "logs") as open_path:
                confirmed = router.handle("confirm")

        self.assertEqual(staged.output, "Open logs folder? Say Confirm Or Deny.")
        open_path.assert_called_once_with(root / "logs", create_directory=True)
        self.assertIn("Opened logs folder", confirmed.output)

    def test_router_stages_and_opens_memory_file_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory_path = root / "data" / "memory.jsonl"
            memory_path.parent.mkdir()
            memory_path.write_text("", encoding="utf-8")
            router = _router(root=root, memory=MemoryStore(path=memory_path))

            staged = router.handle("/open memory")
            with patch("zordon.commands.open_path", return_value=memory_path) as open_path:
                confirmed = router.handle("confirm")

        self.assertEqual(staged.output, "Open memory file? Say Confirm Or Deny.")
        open_path.assert_called_once_with(memory_path, create_directory=False)
        self.assertIn("Opened memory file", confirmed.output)

    def test_router_stages_and_opens_latest_draft_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drafts = root / "drafts"
            drafts.mkdir()
            draft = drafts / "latest.md"
            draft.write_text("Ready.", encoding="utf-8")
            router = _router(root=root)

            staged = router.handle("/open latest draft")
            with patch("zordon.commands.open_path", return_value=draft) as open_path:
                confirmed = router.handle("confirm")

        self.assertEqual(staged.output, "Open latest draft latest.md? Say Confirm Or Deny.")
        open_path.assert_called_once_with(draft, create_directory=False)
        self.assertIn("Opened latest draft", confirmed.output)

    def test_router_reports_voice_setup_command(self):
        outcome = _router(root=Path("C:/Project/Zordon")).handle("/setup-voice")

        self.assertTrue(outcome.handled)
        self.assertIn("Voice setup:", outcome.output)
        self.assertIn("requirements-voice.txt", outcome.output)
        self.assertIn("pip install -r", outcome.output)

    def test_router_lists_and_shows_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text("first\nsecond\n", encoding="utf-8")
            router = _router(root=root)

            listed = router.handle("/logs")
            shown = router.handle("/show-log zordon-2026-07-21.jsonl")

        self.assertTrue(listed.handled)
        self.assertIn("zordon-2026-07-21.jsonl", listed.output)
        self.assertEqual(shown.output, "first\nsecond")

    def test_router_summarizes_log_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text(
                '{"event":"user_turn"}\n{"event":"user_turn"}\n',
                encoding="utf-8",
            )
            router = _router(root=root)

            outcome = router.handle("/events")

        self.assertTrue(outcome.handled)
        self.assertIn("Events in zordon-2026-07-21.jsonl:", outcome.output)
        self.assertIn("- user_turn: 2", outcome.output)

    def test_router_shows_last_log_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text(
                '{"event":"user_turn","text":"Hello"}\n{"event":"assistant_reply","text":"Hi"}\n',
                encoding="utf-8",
            )
            router = _router(root=root)

            outcome = router.handle("/last")

        self.assertTrue(outcome.handled)
        self.assertIn("User: Hello", outcome.output)
        self.assertIn("Assistant: Hi", outcome.output)

    def test_router_searches_latest_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text(
                '{"event":"assistant_reply","text":"Here is the launch hook"}\n',
                encoding="utf-8",
            )
            router = _router(root=root)

            outcome = router.handle("/search-log launch")

        self.assertTrue(outcome.handled)
        self.assertIn("Matches for 'launch'", outcome.output)
        self.assertIn("assistant_reply", outcome.output)

    def test_router_search_log_requires_term(self):
        outcome = _router().handle("/search-log")

        self.assertTrue(outcome.handled)
        self.assertEqual(
            outcome.output, "Usage: /search-log <term> or /search-log <filename> <term>"
        )

    def test_router_show_log_requires_filename(self):
        outcome = _router().handle("/show-log")

        self.assertTrue(outcome.handled)
        self.assertEqual(outcome.output, "Usage: /show-log <filename>")


def _router(
    root: Path | None = None, audit: AuditLog | None = None, memory: MemoryStore | None = None
) -> CommandRouter:
    root = root or Path(".")
    config = Config(
        assistant_name="Zordon",
        model="test-model",
        transcription_model="test-transcribe",
        tts_model="test-tts",
        tts_voice="verse",
        push_to_talk_key="space",
        audio_sample_rate=16000,
        temperature=0.7,
        request_timeout_seconds=5,
    )
    memory = memory or MemoryStore(entries=[])
    audit = audit or AuditLog(events=[])
    agent = Agent(config=config, provider=NoopProvider(), audit=audit, memory=memory)
    return CommandRouter(
        CommandContext(
            config=config,
            agent=agent,
            memory=memory,
            audit=audit,
            root=root,
        )
    )


if __name__ == "__main__":
    unittest.main()

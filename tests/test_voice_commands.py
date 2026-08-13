import unittest

from zordon.voice_cli import normalize_spoken_command


class VoiceCommandNormalizationTest(unittest.TestCase):
    def test_normalizes_spoken_slash_confirm(self):
        self.assertEqual(normalize_spoken_command("Forward slash confirm"), "/confirm")
        self.assertEqual(normalize_spoken_command("slash cancel."), "/cancel")

    def test_normalizes_common_voice_commands(self):
        self.assertEqual(normalize_spoken_command("confirm"), "confirm")
        self.assertEqual(normalize_spoken_command("list drafts"), "/drafts")
        self.assertEqual(normalize_spoken_command("read my latest draft"), "/latest-draft")
        self.assertEqual(normalize_spoken_command("Forward slash images"), "/images")
        self.assertEqual(normalize_spoken_command("Slash commands"), "/commands")
        self.assertEqual(normalize_spoken_command("Slashcommands"), "/commands")
        self.assertEqual(normalize_spoken_command("Slash command"), "/commands")
        self.assertEqual(normalize_spoken_command("List command."), "/commands")
        self.assertEqual(normalize_spoken_command("Slash state"), "/state")
        self.assertEqual(normalize_spoken_command("Slash open state"), "/open-state")
        self.assertEqual(normalize_spoken_command("Slashopenstate"), "/open-state")
        self.assertEqual(normalize_spoken_command("Slash open drafts"), "/open-drafts")
        self.assertEqual(normalize_spoken_command("Slashopendrafts"), "/open-drafts")
        self.assertEqual(normalize_spoken_command("Slash open images"), "/open-images")
        self.assertEqual(normalize_spoken_command("Slashopenimages"), "/open-images")
        self.assertEqual(normalize_spoken_command("Slash open logs"), "/open-logs")
        self.assertEqual(normalize_spoken_command("Slashopenlogs"), "/open-logs")
        self.assertEqual(normalize_spoken_command("Slash open memory"), "/open-memory")
        self.assertEqual(normalize_spoken_command("Slashopenmemory"), "/open-memory")
        self.assertEqual(normalize_spoken_command("Slash open latest draft"), "/open-latest-draft")
        self.assertEqual(normalize_spoken_command("Slashopenlatestdraft"), "/open-latest-draft")
        self.assertEqual(normalize_spoken_command("Slash today"), "/today")
        self.assertEqual(normalize_spoken_command("Slashtoday"), "/today")
        self.assertEqual(normalize_spoken_command("today summary"), "/today")
        self.assertEqual(normalize_spoken_command("Slash ready"), "/ready")
        self.assertEqual(normalize_spoken_command("Slashready"), "/ready")
        self.assertEqual(normalize_spoken_command("Slash recent"), "/recent")
        self.assertEqual(normalize_spoken_command("Slashrecent"), "/recent")
        self.assertEqual(normalize_spoken_command("Slash project"), "/project")
        self.assertEqual(normalize_spoken_command("Slash project status"), "/project-status")
        self.assertEqual(normalize_spoken_command("Slashprojectstatus"), "/project-status")
        self.assertEqual(normalize_spoken_command("Slash projects"), "/projects")
        self.assertEqual(normalize_spoken_command("Flash projects"), "/projects")
        self.assertEqual(normalize_spoken_command("Slash open project"), "/open-project")
        self.assertEqual(
            normalize_spoken_command("New project, Zordon Mixtape."), "/new-project zordon mixtape"
        )
        self.assertEqual(
            normalize_spoken_command("New project Zordon Mixtape"), "/new-project zordon mixtape"
        )
        self.assertEqual(
            normalize_spoken_command("Use project Zordon Mixtape"), "/use-project zordon mixtape"
        )
        self.assertEqual(normalize_spoken_command("Slash project drafts"), "/project-drafts")
        self.assertEqual(
            normalize_spoken_command("Slash open project drafts"), "/open-project-drafts"
        )
        self.assertEqual(
            normalize_spoken_command("Slash save latest draft to project"),
            "/save-latest-draft-to-project",
        )
        self.assertEqual(
            normalize_spoken_command("Slash open latest project draft"),
            "/open-latest-project-draft",
        )
        self.assertEqual(normalize_spoken_command("Slash project images"), "/project-images")
        self.assertEqual(
            normalize_spoken_command("Slash open project images"), "/open-project-images"
        )
        self.assertEqual(
            normalize_spoken_command("Slash save latest image to project"),
            "/save-latest-image-to-project",
        )
        self.assertEqual(
            normalize_spoken_command("Slash open latest project image"),
            "/open-latest-project-image",
        )
        self.assertEqual(normalize_spoken_command("Slash project brief"), "/project-brief")
        self.assertEqual(
            normalize_spoken_command("Slash open project brief"), "/open-project-brief"
        )
        self.assertEqual(
            normalize_spoken_command("Slash project brief next"), "/project-brief-next"
        )
        self.assertEqual(normalize_spoken_command("Slashprojectbriefnext"), "/project-brief-next")
        self.assertEqual(normalize_spoken_command("Project brief next"), "/project-brief-next")
        self.assertEqual(normalize_spoken_command("Slash content plan"), "/content-plan")
        self.assertEqual(normalize_spoken_command("Slashcontentplan"), "/content-plan")
        self.assertEqual(normalize_spoken_command("Make content plan"), "/content-plan")
        self.assertEqual(normalize_spoken_command("Save content plan"), "/save-content-plan")
        self.assertEqual(normalize_spoken_command("Slashsavecontentplan"), "/save-content-plan")
        self.assertEqual(normalize_spoken_command("Open content plan"), "/open-content-plan")
        self.assertEqual(normalize_spoken_command("Slashopencontentplan"), "/open-content-plan")
        self.assertEqual(normalize_spoken_command("Slash project next"), "/project-next")
        self.assertEqual(normalize_spoken_command("Slashprojectnext"), "/project-next")
        self.assertEqual(normalize_spoken_command("Project next"), "/project-next")
        self.assertEqual(
            normalize_spoken_command(
                "Set project brief Zordon Mixtape is a green sci-fi music project."
            ),
            "/set-project-brief zordon mixtape is a green sci-fi music project",
        )
        self.assertEqual(normalize_spoken_command("Slash doctor"), "/doctor")
        self.assertEqual(normalize_spoken_command("Slashdoctor"), "/doctor")
        self.assertEqual(normalize_spoken_command("Slash doktor"), "/doctor")
        self.assertEqual(normalize_spoken_command("Slash made up command"), "/made-up-command")
        self.assertEqual(normalize_spoken_command("latest images"), "/latest-image")
        self.assertEqual(normalize_spoken_command("open latest image"), "/open-latest-image")
        self.assertEqual(normalize_spoken_command("what do you remember?"), "/memory")

    def test_normalizes_common_drafts_mishears(self):
        self.assertEqual(normalize_spoken_command("List drachts"), "/drafts")
        self.assertEqual(normalize_spoken_command("Drafs"), "/drafts")
        self.assertEqual(normalize_spoken_command("List giraffes."), "/drafts")

    def test_leaves_normal_speech_unchanged(self):
        self.assertEqual(
            normalize_spoken_command("Save this as a draft Zordon is voice working."),
            "Save this as a draft Zordon is voice working.",
        )
        self.assertEqual(
            normalize_spoken_command("Tell me about giraffes"), "Tell me about giraffes"
        )


if __name__ == "__main__":
    unittest.main()

"""
sample_schedule.py
------------------
Generates 30 days of realistic sample schedule events starting from today.
Each event is a dict with the fields stored in ChromaDB:
  - id          : unique event ID
  - title       : short event name
  - date        : YYYY-MM-DD
  - start_time  : HH:MM (24-h)
  - end_time    : HH:MM (24-h)
  - type        : meeting | workshop | task | appointment
  - location    : room / URL / None
  - description : longer text used for semantic search
  - attendees   : comma-separated names (optional)
"""

from datetime import date, timedelta


def generate_sample_events(start_date: date | None = None) -> list[dict]:
    """Return a list of sample schedule event dicts for the next 30 days."""
    if start_date is None:
        start_date = date.today()

    events: list[dict] = []
    eid = 1  # simple sequential ID counter

    def add(
        offset: int,
        title: str,
        start: str,
        end: str,
        etype: str,
        location: str = "",
        description: str = "",
        attendees: str = "",
    ):
        nonlocal eid
        event_date = start_date + timedelta(days=offset)
        events.append(
            {
                "id": f"evt_{eid:03d}",
                "title": title,
                "date": event_date.isoformat(),
                "start_time": start,
                "end_time": end,
                "type": etype,
                "location": location,
                "description": description or title,
                "attendees": attendees,
            }
        )
        eid += 1

    # ── Day 0 (Today) ────────────────────────────────────────────────────────
    add(0, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up to sync on progress and blockers.",
        "Alice, Bob, Charlie")
    add(0, "Review Sprint Backlog", "11:00", "12:00", "task",
        "Desk", "Go through the sprint backlog and reprioritize tickets.")
    add(0, "Lunch with Design Team", "12:30", "13:30", "appointment",
        "Cafeteria", "Informal lunch to discuss upcoming UI redesign.",
        "Diana, Eve")

    # ── Day 1 ────────────────────────────────────────────────────────────────
    add(1, "Product Roadmap Planning", "10:00", "11:30", "meeting",
        "Conference Room A", "Quarterly product roadmap planning session.",
        "Alice, Frank, George")
    add(1, "Write Unit Tests", "14:00", "16:00", "task",
        "Desk", "Write unit tests for the new authentication module.")
    add(1, "Doctor Appointment", "17:00", "17:45", "appointment",
        "City Health Clinic", "Annual check-up with Dr. Smith.")

    # ── Day 2 ────────────────────────────────────────────────────────────────
    add(2, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(2, "Python Best Practices Workshop", "10:00", "13:00", "workshop",
        "Training Room B", "Hands-on workshop covering Python typing, testing, and async patterns.",
        "All Engineering")
    add(2, "Code Review Session", "15:00", "16:00", "meeting",
        "Zoom", "Review PRs for the billing service refactor.",
        "Bob, Charlie, Hannah")

    # ── Day 3 ────────────────────────────────────────────────────────────────
    add(3, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(3, "Data Pipeline Design Review", "11:00", "12:00", "meeting",
        "Conference Room B", "Review proposed architecture for new ETL pipeline.",
        "Ivan, Julia")
    add(3, "Gym Session", "18:00", "19:00", "appointment",
        "FitLife Gym", "Personal training session.")

    # ── Day 4 ────────────────────────────────────────────────────────────────
    add(4, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(4, "Frontend Performance Audit", "10:30", "12:30", "task",
        "Desk", "Audit and optimize Lighthouse scores for the main dashboard.")
    add(4, "1-on-1 with Manager", "14:00", "14:30", "meeting",
        "Manager's Office", "Weekly 1-on-1 check-in with manager Karen.",
        "Karen")

    # ── Day 5 ────────────────────────────────────────────────────────────────
    add(5, "Sprint Retrospective", "10:00", "11:00", "meeting",
        "Zoom", "End-of-sprint retrospective: what went well, what to improve.",
        "Alice, Bob, Charlie, Diana")
    add(5, "Dentist Appointment", "14:30", "15:30", "appointment",
        "Smile Dental Care", "Six-month cleaning and check-up.")

    # ── Day 6 (Weekend Day 1) ─────────────────────────────────────────────────
    add(6, "Side Project Coding", "10:00", "13:00", "task",
        "Home", "Work on personal open-source project — add GraphQL support.")
    add(6, "Coffee with Mentor", "15:00", "16:00", "appointment",
        "Blue Bottle Coffee", "Monthly mentorship coffee chat.", "Mentor Leo")

    # ── Day 7 (Weekend Day 2) ─────────────────────────────────────────────────
    add(7, "Weekly Planning", "09:00", "10:00", "task",
        "Home", "Plan tasks and priorities for the upcoming week.")

    # ── Day 8 ────────────────────────────────────────────────────────────────
    add(8, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(8, "New Hire Onboarding", "10:00", "12:00", "meeting",
        "Conference Room C", "Onboarding session for new engineer Marcus.",
        "Marcus, HR Team")
    add(8, "API Documentation Update", "13:00", "15:00", "task",
        "Desk", "Update REST API documentation in Confluence.")

    # ── Day 9 ────────────────────────────────────────────────────────────────
    add(9, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(9, "ML Model Review", "11:00", "12:30", "meeting",
        "Conference Room A", "Review performance metrics of the recommendation model.",
        "Data Science Team")
    add(9, "Lunch & Learn: Kubernetes", "12:30", "13:30", "workshop",
        "Main Hall", "Intro to Kubernetes for developers — catered lunch included.",
        "All Engineering")

    # ── Day 10 ───────────────────────────────────────────────────────────────
    add(10, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(10, "Security Audit Prep", "14:00", "16:00", "task",
        "Desk", "Prepare documentation and access logs for upcoming security audit.")
    add(10, "Yoga Class", "18:30", "19:30", "appointment",
        "Zen Studio", "Weekly yoga session.")

    # ── Day 11 ───────────────────────────────────────────────────────────────
    add(11, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(11, "Client Demo — Acme Corp", "13:00", "14:00", "meeting",
        "Zoom", "Live product demo for Acme Corp stakeholders.",
        "Client: John Doe, Jane Doe")
    add(11, "Post-Demo Debrief", "14:30", "15:00", "meeting",
        "Conference Room A", "Internal debrief after Acme Corp demo.",
        "Alice, Frank")

    # ── Day 12 ───────────────────────────────────────────────────────────────
    add(12, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(12, "Cloud Cost Optimization Workshop", "10:00", "12:00", "workshop",
        "Training Room A", "Workshop on reducing AWS spend with reserved instances and right-sizing.",
        "DevOps, Engineering Leads")
    add(12, "Gym Session", "18:00", "19:00", "appointment",
        "FitLife Gym", "Personal training session.")

    # ── Day 13 (Weekend) ─────────────────────────────────────────────────────
    add(13, "Family Brunch", "10:30", "12:30", "appointment",
        "Mom's House", "Weekly family brunch.")

    # ── Day 14 (Weekend) ─────────────────────────────────────────────────────
    add(14, "Weekly Planning", "09:00", "10:00", "task",
        "Home", "Plan tasks and priorities for week 3.")

    # ── Day 15 ───────────────────────────────────────────────────────────────
    add(15, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(15, "Quarterly Business Review", "10:00", "12:00", "meeting",
        "Board Room", "Q3 business review with leadership team.",
        "Leadership Team, Alice")
    add(15, "Write Engineering Blog Post", "14:00", "16:00", "task",
        "Desk", "Draft blog post on building scalable microservices with FastAPI.")

    # ── Day 16 ───────────────────────────────────────────────────────────────
    add(16, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(16, "Database Schema Migration", "10:00", "13:00", "task",
        "Desk", "Execute and verify database schema migration for v2.0 release.")
    add(16, "1-on-1 with Manager", "15:00", "15:30", "meeting",
        "Manager's Office", "Bi-weekly 1-on-1 with Karen.", "Karen")

    # ── Day 17 ───────────────────────────────────────────────────────────────
    add(17, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(17, "DevOps Sync", "11:00", "11:45", "meeting",
        "Conference Room B", "Sync with DevOps team on CI/CD pipeline improvements.",
        "Ivan, DevOps Team")
    add(17, "Yoga Class", "18:30", "19:30", "appointment",
        "Zen Studio", "Weekly yoga session.")

    # ── Day 18 ───────────────────────────────────────────────────────────────
    add(18, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(18, "Performance Review Prep", "13:00", "15:00", "task",
        "Desk", "Prepare self-evaluation document for upcoming performance review cycle.")
    add(18, "Happy Hour", "17:00", "19:00", "appointment",
        "The Local Bar", "Team happy hour to celebrate sprint completion.",
        "Whole Team")

    # ── Day 19 (Weekend) ─────────────────────────────────────────────────────
    add(19, "Hackathon — Day 1", "09:00", "18:00", "workshop",
        "Innovation Lab", "Company internal hackathon — Day 1 of 2.",
        "All Volunteers")

    # ── Day 20 (Weekend) ─────────────────────────────────────────────────────
    add(20, "Hackathon — Day 2 + Demo", "09:00", "17:00", "workshop",
        "Innovation Lab", "Company internal hackathon — Day 2, finish and demo projects.",
        "All Volunteers")

    # ── Day 21 ───────────────────────────────────────────────────────────────
    add(21, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(21, "Sprint Planning", "10:00", "12:00", "meeting",
        "Conference Room A", "Sprint planning for the next two-week sprint.",
        "Alice, Bob, Charlie, Diana")
    add(21, "Physio Appointment", "16:00", "17:00", "appointment",
        "ActiveCare Physio", "Follow-up physiotherapy session.")

    # ── Day 22 ───────────────────────────────────────────────────────────────
    add(22, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(22, "Advanced RAG Workshop", "10:00", "13:00", "workshop",
        "Training Room B", "Deep dive into retrieval-augmented generation patterns and evaluation.",
        "AI/ML Team")
    add(22, "Gym Session", "18:00", "19:00", "appointment",
        "FitLife Gym", "Personal training session.")

    # ── Day 23 ───────────────────────────────────────────────────────────────
    add(23, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(23, "Investor Update Call", "14:00", "15:00", "meeting",
        "Zoom", "Monthly investor update call — demo new features.",
        "CEO, CFO, Lead Investors")

    # ── Day 24 ───────────────────────────────────────────────────────────────
    add(24, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(24, "Release v2.0 Deployment", "10:00", "12:00", "task",
        "Desk", "Deploy v2.0 to production and monitor error rates / dashboards.")
    add(24, "Post-Release Monitoring", "14:00", "16:00", "task",
        "Desk", "Monitor Datadog dashboards and on-call Slack channel after v2.0 release.")

    # ── Day 25 ───────────────────────────────────────────────────────────────
    add(25, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(25, "Architecture Review Board", "11:00", "13:00", "meeting",
        "Board Room", "ARB review for proposed event-driven architecture migration.",
        "Engineering Leads, CTO")
    add(25, "Yoga Class", "18:30", "19:30", "appointment",
        "Zen Studio", "Weekly yoga session.")

    # ── Day 26 (Weekend) ─────────────────────────────────────────────────────
    add(26, "Conference Talk Prep", "10:00", "13:00", "task",
        "Home", "Finalize slides and demo for upcoming tech conference talk.")

    # ── Day 27 (Weekend) ─────────────────────────────────────────────────────
    add(27, "Family Brunch", "10:30", "12:30", "appointment",
        "Mom's House", "Weekly family brunch.")

    # ── Day 28 ───────────────────────────────────────────────────────────────
    add(28, "Morning Stand-up", "09:00", "09:30", "meeting",
        "Zoom", "Daily team stand-up.")
    add(28, "Tech Conference — Day 1", "09:00", "18:00", "workshop",
        "Convention Center", "Attending / presenting at the annual tech conference.",
        "Alice, Frank, External Speakers")

    # ── Day 29 ───────────────────────────────────────────────────────────────
    add(29, "Tech Conference — Day 2", "09:00", "17:00", "workshop",
        "Convention Center", "Day 2 of the annual tech conference — networking and workshops.",
        "Alice, Frank, External Speakers")

    return events


if __name__ == "__main__":
    events = generate_sample_events()
    print(f"Generated {len(events)} sample events over 30 days.\n")
    for e in events[:5]:
        print(e)

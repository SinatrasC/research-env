"""mini-calendar: research-env's General Agent task shape (a small DB, stateful tools, a gold action
sequence and verify(db)) as a verifiers v1 taskset.

`variant=base` has three tasks: t0 is valid, t1's gold books the wrong slot (vf-validate's gold check
flags it), t2 is already solved before any action (the noop check flags it). `variant=evolved` repairs
both and adds two tasks whose requested slot is not the first free one, so an agent that books the
first free slot fails them. Task ids are stable across variants, so runs pair task by task.
"""

from typing import Literal

import verifiers.v1 as vf

from mini_calendar.servers.calendar import CalendarState, CalendarToolset


class CalendarData(vf.TaskData):
    slots: dict[str, str | None]
    person: str
    want: str
    gold: list[tuple[str, dict[str, str]]]


class CalendarTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


def replay(slots: dict[str, str | None], gold) -> dict[str, str | None] | None:
    """The DB after the gold actions, or None if an action is invalid."""
    db = dict(slots)
    for name, kwargs in gold:
        if name != "book_slot" or db.get(kwargs["slot"], "missing") is not None:
            return None
        db[kwargs["slot"]] = kwargs["person"]
    return db


class CalendarTask(vf.Task[CalendarData, CalendarState, CalendarTaskConfig]):
    @classmethod
    def toolsets(cls, config: CalendarTaskConfig) -> list[vf.Toolset]:
        return [CalendarToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.slots = dict(self.data.slots)

    @vf.reward
    async def booked(self, trace: vf.Trace) -> float:
        return float(trace.state.slots.get(self.data.want) == self.data.person)

    async def validate(self, runtime: vf.Runtime) -> bool:
        db = replay(self.data.slots, self.data.gold)
        return db is not None and db.get(self.data.want) == self.data.person


def _book(slot: str, person: str) -> list[tuple[str, dict[str, str]]]:
    return [("book_slot", {"slot": slot, "person": person})]


# (id, slots, person, wanted slot, gold actions)
BASE = [
    ("t0", {"09:00": None, "10:00": "bob"}, "ana", "09:00", _book("09:00", "ana")),
    ("t1", {"09:00": None, "10:00": None}, "ana", "09:00", _book("10:00", "ana")),  # bad gold
    ("t2", {"09:00": "ana", "10:00": None}, "ana", "09:00", _book("10:00", "ana")),  # pre-solved
]
EVOLVED = [
    BASE[0],
    ("t1", {"09:00": None, "10:00": None}, "ana", "10:00", _book("10:00", "ana")),
    ("t2", {"09:00": "bob", "10:00": None, "11:00": None}, "ana", "11:00", _book("11:00", "ana")),
    ("t3", {"09:00": None, "11:00": "cem"}, "dan", "09:00", _book("09:00", "dan")),
    ("t4", {"09:00": "bob", "10:00": None}, "eve", "10:00", _book("10:00", "eve")),
]


class MiniCalendarConfig(vf.TasksetConfig):
    variant: Literal["base", "evolved"] = "base"
    task: CalendarTaskConfig = CalendarTaskConfig()


class MiniCalendarTaskset(vf.Taskset[CalendarTask, MiniCalendarConfig]):
    def load(self) -> list[CalendarTask]:
        rows = BASE if self.config.variant == "base" else EVOLVED
        return [
            CalendarTask(
                CalendarData(
                    id=task_id,
                    prompt=f"Book the {want} slot for {person} using the calendar tools.",
                    slots=slots,
                    person=person,
                    want=want,
                    gold=gold,
                ),
                self.config.task,
            )
            for task_id, slots, person, want, gold in rows
        ]

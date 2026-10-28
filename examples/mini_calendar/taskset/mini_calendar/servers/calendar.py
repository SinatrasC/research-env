"""The calendar tools, served over MCP per rollout. Tools appear to the agent as
`calendar_list_slots` and `calendar_book_slot`; both read and write the rollout's state."""

import verifiers.v1 as vf


class CalendarState(vf.State):
    slots: dict[str, str | None] = {}


class CalendarToolset(vf.Toolset[vf.ToolsetConfig, CalendarState]):
    TOOL_PREFIX = "calendar"

    @vf.tool
    def list_slots(self) -> dict[str, str | None]:
        """Every slot and who holds it (null = free)."""
        return self.state.slots

    @vf.tool
    def book_slot(self, slot: str, person: str) -> str:
        """Book a free slot for a person."""
        if self.state.slots.get(slot, "missing") is not None:
            return f"error: {slot} is not free"
        self.state.slots = {**self.state.slots, slot: person}
        return f"booked {slot} for {person}"


if __name__ == "__main__":
    CalendarToolset.run()

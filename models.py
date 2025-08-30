from sqlmodel import SQLModel, Field, Session, Relationship, create_engine, select
from typing import Optional, List, Any, Union
from pydantic import PrivateAttr
from enum import Enum
import json
import simpy
import uuid
from datetime import datetime


class Patient(SQLModel, table=True):
    """Track practitioners/system actors who make changes"""

    resource_type: str = "Patient"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    first_name: str
    last_name: str
    gender: str
    birthdate: datetime


class Practitioner(SQLModel, table=True):
    """Combined SQLModel and simulation Practitioner class"""

    # Database fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    resource_type: str = "Practitioner"
    first_name: str
    last_name: str
    gender: str
    birthdate: datetime
    role: str = "doctor"  # default role

    # Relationships
    changes: List["Provenance"] = Relationship(back_populates="practitioner")

    # Simulation-specific fields (not persisted in database)
    _env: Any = PrivateAttr(default=None)
    _resource: Any = PrivateAttr(default=None)
    _work_schedule: Union[dict, None] = PrivateAttr(default=None)

    def __init__(self, env=None, work_schedule=None, **data):
        super().__init__(**data)

        # Handle simulation initialization
        if env is not None:
            self._init_simulation(env, work_schedule)

    def _init_simulation(self, env, work_schedule):
        """Initialize simulation-specific components"""
        self._env = env
        self._resource = simpy.Resource(env, capacity=1)
        self._work_schedule = work_schedule or {
            i: [(9 * 60, 17 * 60)] for i in range(5)
        }

    # Simulation methods
    def is_within_working_hours(self, duration):
        now = self._env.now
        weekday = int((now // (24 * 60)) % 7)
        minute_of_day = int(now % (24 * 60))

        for start, end in self._work_schedule.get(weekday, []):
            if start <= minute_of_day <= end - duration:
                return True
        return False

    def can_take_appointment(self, duration):
        return self.is_within_working_hours(duration)

    @property
    def name(self):
        """Get full name of the practitioner"""
        return f"{self.first_name} {self.last_name} ({self.id})"

    # Also provide access to simulation resources
    @property
    def env(self):
        return self._env

    @property
    def resource(self):
        return self._resource

    @property
    def work_schedule(self):
        return self._work_schedule


class Provenance(SQLModel, table=True):
    """Track who changed what and when"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    action: str  # "create", "update", "delete", "access"
    recorded: int
    target_resource_type: str  # "Appointment", "Encounter", etc.
    target_resource_id: str  # ID of the affected resource
    practitioner_id: int = Field(foreign_key="practitioner.id")
    practitioner: Practitioner = Relationship(back_populates="changes")


class FHIRBase(SQLModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    resource_type: str
    patient_id: str
    practitioner_id: Optional[str] = None


class AppointmentStatus(str, Enum):
    # PROPOSED = "proposed"
    # PENDING = "pending"
    BOOKED = "booked"
    # FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    NOSHOW = "noshow"
    # IN_PROGRESS = "in-progress"
    FINISHED = "finished"


class Appointment(FHIRBase, table=True):
    resource_type: str = "Appointment"
    created: int
    scheduled_start_time: int  # When it was scheduled to occur
    duration: int
    status: AppointmentStatus
    cancellation_reason: Optional[str] = None

    @property
    def start(self):
        return self.scheduled_start_time

    @property
    def end(self):
        return self.scheduled_start_time + self.duration


class Encounter(FHIRBase, table=True):
    resource_type: str = "Encounter"
    appointment_id: str | None = Field(default=None, foreign_key="appointment.id")
    actual_start_time: int
    duration: int

    @property
    def start(self):
        return self.actual_start_time

    @property
    def end(self):
        return self.actual_start_time + self.duration


class Observation(FHIRBase, table=True):
    resource_type: str = "Observation"
    encounter_id: str | None = Field(default=None, foreign_key="encounter.id")
    code: str
    value: Optional[str] = None
    timestamp: int


class AccessEventType(str, Enum):
    EMERGENCY = "emergency-access"
    CARE = "care-access"


class AccessEventPurpose(str, Enum):
    # Concepts from: https://terminology.hl7.org/5.1.0/ValueSet-v3-PurposeOfUse.html
    EMERGENCY = "BTG"
    CARE = "CAREMGT"


class AuditEvent(FHIRBase, table=True):
    resource_type: str = Field(default="AuditEvent", const=True)

    # Minimal required FHIR AuditEvent fields
    recorded: int  # When the event was recorded (simulation time)
    action: str = Field(default="R")  # R=Read, C=Create, U=Update, D=Delete, E=Execute

    # Optional target resource details (can be None for general system access)
    # Otherwise, e.g., "Appointment", "Encounter", "Observation"
    target_resource_type: Optional[str] = None
    target_resource_id: Optional[str] = None

    # FHIR-aligned categorization

    # The 'event_type' is e.g., 'care-access' or "emergency-access"
    event_type: str = Field(default=AccessEventType.CARE)
    # The purpose of athorization is e.g., (with respect to 'event_type')
    # "CAREMGT" or "CAREMGT"
    purpose: str = Field(default=AccessEventPurpose.CARE)

    # Outcome information
    outcome: str = Field(default="success")  # success | failure

    # Relationships
    # practitioner_id: Who performed the action
    practitioner_id: str = Field(foreign_key="practitioner.id")
    # Which patient's data was accessed (optional)
    patient_id: str = Field(foreign_key="patient.id")

    # Optional free-text explanation
    purpose_of_event: Optional[str] = None


class ProvenanceTracker:
    def __init__(self, engine):
        self.engine = engine

    def _get_practitioner(self, practitioner_id: str) -> Practitioner:
        with Session(self.engine) as session:
            practitioner = session.exec(
                select(Practitioner).where(Practitioner.id == practitioner_id)
            ).first()
            if not practitioner:
                raise ValueError("Practitioner not found")
            return practitioner

    def log_change(
        self,
        action: str,
        target_resource_type: str,
        target_resource_id: str,
        recorded: int,
        practitioner: Practitioner,
        before: Optional[dict] = None,
        after: Optional[dict] = None,
    ):
        with Session(self.engine) as session:
            prov = Provenance(
                action=action,
                target_resource_type=target_resource_type,
                target_resource_id=target_resource_id,
                practitioner_id=practitioner.id,
                recorded=recorded,
                details=(
                    json.dumps({"before": before, "after": after})
                    if before or after
                    else None
                ),
            )
            session.add(prov)
            session.commit()


class FHIRLogger:
    def __init__(self, provenance_tracker: ProvenanceTracker, engine):
        self.provenance = provenance_tracker
        self.engine = engine

    def _log_provenance(
        self,
        action: str,
        recorded: int,
        resource: FHIRBase,
        practitioner: Practitioner,
        before: Optional[dict] = None,
        after: Optional[dict] = None,
    ):
        """Log a provenance entry with before/after states"""
        self.provenance.log_change(
            action=action,
            target_resource_type=resource.resource_type,
            target_resource_id=resource.id,
            practitioner=practitioner,
            before=before,
            after=after,
            recorded=recorded,
        )

    def log_appointment(
        self,
        patient_id: str,
        created: int,
        status: AppointmentStatus,
        practitioner_id: str,
        duration: int,
        scheduled_start_time: int,
        cancellation_reason: Optional[str] = None,
    ):
        with Session(self.engine) as session:
            # Get practitioner user
            practitioner_object = self.provenance._get_practitioner(practitioner_id)

            appointment = Appointment(
                patient_id=patient_id,
                created=created,
                status=status,
                practitioner_id=practitioner_id,
                duration=duration,
                scheduled_start_time=scheduled_start_time,
                cancellation_reason=cancellation_reason,
            )
            session.add(appointment)
            session.commit()
            session.refresh(appointment)

            # Log creation
            self._log_provenance(
                action="create",
                recorded=created,
                resource=appointment,
                practitioner=practitioner_object,
            )

            return appointment.id

    def update_appointment_status(
        self,
        appointment_id: str,
        new_status: AppointmentStatus,
        recorded: int,
        practitioner_id: str,
        reason: Optional[str] = None,
    ):
        with Session(self.engine) as session:
            appointment_object = session.get(Appointment, appointment_id)
            if not appointment_object:
                raise ValueError("Appointment not found")

            # Get before state
            before_state = {
                "status": appointment_object.status,
                "cancellation_reason": appointment_object.cancellation_reason,
            }

            # Update
            appointment_object.status = new_status
            if reason:
                appointment_object.cancellation_reason = reason

            # Get practitioner user
            practitioner_object = self.provenance._get_practitioner(practitioner_id)

            session.add(appointment_object)
            session.commit()
            session.refresh(appointment_object)

            # Log update
            self._log_provenance(
                action="update",
                recorded=recorded,
                resource=appointment_object,
                practitioner=practitioner_object,
                before=before_state,
                after={
                    "status": appointment_object.status,
                    "cancellation_reason": appointment_object.cancellation_reason,
                },
            )

    def log_encounter(
        self,
        patient_id: str,
        practitioner_id: str,
        actual_start_time: int,
        duration: int,
        appointment_id: Optional[str] = None,
    ):
        with Session(self.engine) as session:
            encounter = Encounter(
                patient_id=patient_id,
                practitioner_id=practitioner_id,
                appointment_id=appointment_id,
                actual_start_time=actual_start_time,
                duration=duration,
            )
            session.add(encounter)
            session.commit()
            return encounter.id

    def log_observation(
        self,
        patient_id: str,
        practitioner_id: str,
        timestamp: int,
        code: str,
        value: Optional[str] = None,
        encounter_id: Optional[str] = None,
    ):
        with Session(self.engine) as session:
            obs = Observation(
                patient_id=patient_id,
                practitioner_id=practitioner_id,
                timestamp=timestamp,
                code=code,
                value=value,
                encounter_id=encounter_id,
            )
            session.add(obs)
            session.commit()

    def log_access_event(
        self,
        patient_id: str,
        recorded: int,
        practitioner_id: str,
        action: str,
        event_type: AccessEventType,
        purpose: AccessEventPurpose,
        purpose_of_event: Optional[str] = None,
        target_resource_type: Optional[str] = None,
        target_resource_id: Optional[str] = None,
        outcome: str = "success",
    ):
        """Log an access event (possibly with context)"""
        with Session(self.engine) as session:
            # Get practitioner and patient objects
            practitioner = session.get(Practitioner, practitioner_id)
            patient = session.get(Patient, patient_id)

            if not practitioner or not patient:
                raise ValueError("Practitioner or Patient not found")

            # Create the  event
            audit_event = AuditEvent(
                event_type=event_type,
                recorded=recorded,
                target_resource_type=target_resource_type,
                target_resource_id=target_resource_id,
                action=action,
                purpose=purpose,
                purpose_of_event=purpose_of_event,
                outcome=outcome,
                practitioner_id=practitioner_id,
                patient_id=patient_id,
            )

            session.add(audit_event)
            session.commit()
            session.refresh(audit_event)

            # Log provenance
            self._log_provenance(
                action="create",
                recorded=recorded,
                resource=audit_event,
                practitioner=practitioner,
                before=None,
                after={
                    "event_type": event_type,
                    "purpose": purpose,
                    "target_resource": (
                        f"{target_resource_type}/{target_resource_id}"
                        if target_resource_type
                        else None
                    ),
                    "outcome": outcome,
                },
            )

            return audit_event.id

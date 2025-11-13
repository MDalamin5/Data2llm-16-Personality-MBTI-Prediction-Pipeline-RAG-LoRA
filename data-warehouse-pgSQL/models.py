# models.py
from sqlmodel import SQLModel, Field, Relationship, Column, ARRAY, String, CheckConstraint
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy.types import TIMESTAMP, Numeric
from sqlalchemy import text, func

class CompanySearch(SQLModel, table=True):
    __tablename__ = "company_searches"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    search_keyword: str = Field(nullable=False)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    company_links: List["CompanyLink"] = Relationship(back_populates="company_search")

class CompanyLink(SQLModel, table=True):
    __tablename__ = "company_links"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    company_search_id: UUID = Field(foreign_key="company_searches.id", nullable=False)
    website_link: str = Field(nullable=False)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    company_search: "CompanySearch" = Relationship(back_populates="company_links")
    company_details: List["CompanyDetail"] = Relationship(back_populates="company_link")

class CompanyDetail(SQLModel, table=True):
    __tablename__ = "company_details"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    company_link_id: UUID = Field(foreign_key="company_links.id", nullable=False)
    company_name: Optional[str] = Field(default=None)
    linkedin_link: Optional[str] = Field(default=None)
    facebook_link: Optional[str] = Field(default=None)
    x_link: Optional[str] = Field(default=None)
    instagram_link: Optional[str] = Field(default=None)
    youtube_link: Optional[str] = Field(default=None)
    other_links: Optional[str] = Field(default=None)
    phones: Optional[List[str]] = Field(sa_column=Column(ARRAY(String())))
    about_page_text: Optional[str] = Field(default=None)
    contact_page_text: Optional[str] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    company_link: "CompanyLink" = Relationship(back_populates="company_details")
    company_employees: List["CompanyEmployee"] = Relationship(back_populates="company_detail")

class CompanyEmployee(SQLModel, table=True):
    __tablename__ = "company_employees"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    company_detail_id: UUID = Field(foreign_key="company_details.id", nullable=False)
    name: str = Field(nullable=False)
    designation_or_headline: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    linkedin_profile_link: str = Field(nullable=False, unique=True)
    other_info: Optional[str] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    company_detail: "CompanyDetail" = Relationship(back_populates="company_employees")
    employee_detail: Optional["EmployeeDetail"] = Relationship(back_populates="company_employee")

# class EmployeeDetail(SQLModel, table=True):
#     __tablename__ = "employee_details"
#     id: UUID = Field(default_factory=uuid4, primary_key=True)
#     company_employee_id: UUID = Field(foreign_key="company_employees.id", nullable=False)
#     name: Optional[str] = Field(default=None)
#     headline: Optional[str] = Field(default=None)
#     location: Optional[str] = Field(default=None)
#     contact_info_url: Optional[str] = Field(default=None)
#     public_profile_url: Optional[str] = Field(default=None)
#     about: Optional[str] = Field(default=None)
#     current_company_name: Optional[str] = Field(default=None)
#     is_deleted: bool = Field(default=False)
#     created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
#     updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
#     company_employee: "CompanyEmployee" = Relationship(back_populates="employee_detail")
#     connection_details: List["ConnectionDetail"] = Relationship(back_populates="employee_detail")
#     featured_sections: List["FeaturedSection"] = Relationship(back_populates="employee_detail")
#     skills: List["Skill"] = Relationship(back_populates="employee_detail")
#     licenses_certifications: List["LicenseCertification"] = Relationship(back_populates="employee_detail")
#     experiences: List["Experience"] = Relationship(back_populates="employee_detail")
#     educations: List["Education"] = Relationship(back_populates="employee_detail")
#     top_voices: List["TopVoice"] = Relationship(back_populates="employee_detail")

# ---> Update One <----

class EmployeeDetail(SQLModel, table=True):
    __tablename__ = "employee_details"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    company_employee_id: UUID = Field(foreign_key="company_employees.id", nullable=False)

    name: Optional[str] = Field(default=None)
    headline: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    contact_info_url: Optional[str] = Field(default=None)
    public_profile_url: Optional[str] = Field(default=None)
    about: Optional[str] = Field(default=None)

    # New addition
    company_name: Optional[str] = Field(default=None)

    current_company_name: Optional[str] = Field(default=None)

    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(
        sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(
        sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"),
                         onupdate=func.current_timestamp(), nullable=False))

    company_employee: "CompanyEmployee" = Relationship(back_populates="employee_detail")
    connection_details: List["ConnectionDetail"] = Relationship(back_populates="employee_detail")
    featured_sections: List["FeaturedSection"] = Relationship(back_populates="employee_detail")
    skills: List["Skill"] = Relationship(back_populates="employee_detail")
    licenses_certifications: List["LicenseCertification"] = Relationship(back_populates="employee_detail")
    experiences: List["Experience"] = Relationship(back_populates="employee_detail")
    educations: List["Education"] = Relationship(back_populates="employee_detail")
    top_voices: List["TopVoice"] = Relationship(back_populates="employee_detail")
    friends_and_family: List["FriendsAndFamily"] = Relationship(back_populates="employee_detail")





class FriendsAndFamily(SQLModel, table=True):
    __tablename__ = "friends_and_family"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    employee_detail_id: UUID = Field(foreign_key="employee_details.id", nullable=False)

    # NEW field
    name: Optional[str] = Field(default=None)
    # existing fields
    profession: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    designation_title: Optional[str] = Field(default=None)
    employee_with_relation: Optional[str] = Field(default=None)

    

    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(
        sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    )
    updated_at: Optional[datetime] = Field(
        sa_column=Column(
            TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"),
            onupdate=func.current_timestamp(), nullable=False
        )
    )

    employee_detail: "EmployeeDetail" = Relationship(back_populates="friends_and_family")





class ConnectionDetail(SQLModel, table=True):
    __tablename__ = "connection_details"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    employee_detail_id: UUID = Field(foreign_key="employee_details.id", nullable=False)
    connection_degree: Optional[str] = Field(default=None)
    total_connections: Optional[int] = Field(default=None)
    mutual_connections: Optional[int] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    employee_detail: "EmployeeDetail" = Relationship(back_populates="connection_details")

class FeaturedSection(SQLModel, table=True):
    __tablename__ = "featured_sections"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    employee_detail_id: UUID = Field(foreign_key="employee_details.id", nullable=False)
    type: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    link: Optional[str] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    employee_detail: "EmployeeDetail" = Relationship(back_populates="featured_sections")

class Skill(SQLModel, table=True):
    __tablename__ = "skills"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    employee_detail_id: UUID = Field(foreign_key="employee_details.id", nullable=False)
    # skill_title: str = Field(nullable=False)
    skill_title: Optional[str] = Field(default=None)
    where_used: Optional[str] = Field(default=None)
    endorsed_by: Optional[str] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    employee_detail: "EmployeeDetail" = Relationship(back_populates="skills")

class LicenseCertification(SQLModel, table=True):
    __tablename__ = "licenses_certifications"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    employee_detail_id: UUID = Field(foreign_key="employee_details.id", nullable=False)
    # name: str = Field(nullable=False)
    name: Optional[str] = Field(default=None)
    issuing_organization: Optional[str] = Field(default=None)
    issue_date: Optional[datetime] = Field(default=None)
    credential_url: Optional[str] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    employee_detail: "EmployeeDetail" = Relationship(back_populates="licenses_certifications")

class Experience(SQLModel, table=True):
    __tablename__ = "experiences"
    # __table_args__ = (CheckConstraint("work_type IN ('onsite', 'remote', 'hybrid')", name="check_work_type"),)
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    employee_detail_id: UUID = Field(foreign_key="employee_details.id", nullable=False)
    title: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    time_duration: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    work_type: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    employee_detail: "EmployeeDetail" = Relationship(back_populates="experiences")

class Education(SQLModel, table=True):
    __tablename__ = "educations"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    employee_detail_id: UUID = Field(foreign_key="employee_details.id", nullable=False)
    degree: Optional[str] = Field(default=None)
    institution_name: Optional[str] = Field(default=None)
    cgpa: Optional[float] = Field(sa_column=Column(Numeric), default=None)
    skills: Optional[str] = Field(default=None)
    duration: Optional[str] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    employee_detail: "EmployeeDetail" = Relationship(back_populates="educations")

class TopVoice(SQLModel, table=True):
    __tablename__ = "top_voices"
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    employee_detail_id: UUID = Field(foreign_key="employee_details.id", nullable=False)
    profile_name: Optional[str] = Field(default=None)
    headline_designation: Optional[str] = Field(default=None)
    follower_number: Optional[int] = Field(default=None)
    duration: Optional[str] = Field(default=None)
    is_deleted: bool = Field(default=False)
    created_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False))
    updated_at: Optional[datetime] = Field(sa_column=Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=func.current_timestamp(), nullable=False))
    
    employee_detail: "EmployeeDetail" = Relationship(back_populates="top_voices")
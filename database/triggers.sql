
DROP TRIGGER IF EXISTS trg_case_created;
DROP TRIGGER IF EXISTS trg_case_status_changed;

CREATE TRIGGER trg_case_created
AFTER INSERT ON investigation_cases
BEGIN
    INSERT INTO investigation_history(case_id, action, notes)
    VALUES (
        NEW.case_id,
        'case_opened',
        'Investigation case automatically recorded'
    );
END;

CREATE TRIGGER trg_case_status_changed
AFTER UPDATE OF status ON investigation_cases
WHEN OLD.status <> NEW.status
BEGIN
    INSERT INTO investigation_history(case_id, action, notes)
    VALUES (
        NEW.case_id,
        'status_changed',
        'Status changed from ' || OLD.status || ' to ' || NEW.status
    );
END;

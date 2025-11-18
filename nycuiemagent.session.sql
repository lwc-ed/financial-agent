ALTER TABLE credit_card_benefits.ctbc_linepay_benefits
MODIFY COLUMN brands_text TEXT;

ALTER TABLE credit_card_benefits.ctbc_linepay_benefits
DROP COLUMN brands_text;


ALTER TABLE credit_card_benefits.ctbc_linepay_debit_benefits
MODIFY COLUMN brands_text TEXT;

ALTER TABLE credit_card_benefits.ctbc_linepay_debit_benefits
DROP COLUMN brands_text;
ALTER TABLE credit_card_benefits.ctbc_linepay_debit_benefits
ADD COLUMN brands_text VARCHAR(255)
    GENERATED ALWAYS AS (
        REPLACE(
            REPLACE(
                REPLACE(brands, '["', ''),
            '"]', ''),
        '","', ', ')
    ) STORED;
ALTER TABLE credit_card_benefits.ctbc_linepay_benefits
ADD COLUMN brands_text VARCHAR(255)
    GENERATED ALWAYS AS (
        REPLACE(
            REPLACE(
                REPLACE(brands, '["', ''),
            '"]', ''),
        '","', ', ')
    ) STORED;
    
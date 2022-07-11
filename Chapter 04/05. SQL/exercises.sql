-- Exercise 1:

-- SELECT 
--     *
--     FROM film;

-- Exercise 2:

-- SELECT
--     district,
--     phone,
--     postal_code
--     FROM address;
    
-- Exercise 3:

-- SELECT
--     address ||' '|| district ||' '|| postal_code
--     AS full_address FROM address
    
-- Exercise 4:

-- SELECT
--     first_name,
--     last_name
-- FROM
--     customer
-- WHERE
--     first_name LIKE 'J%'
    
-- Exercise 5:

-- SELECT
--     *
-- FROM
--     payment
-- WHERE
--     amount BETWEEN '3' AND '5'
    
-- Exercise 6:

-- SELECT
--     *
-- FROM
--     payment
-- WHERE
--     payment_date BETWEEN '2007-02-15' AND '2077-02-20'
--     -- Extra
-- --     amount < 8 AND payment_date BETWEEN '2007-02-15' AND '2077-02-20'

-- Exercise 7:

-- SELECT
--     title
-- FROM
--     film
-- WHERE
--     film_id IN (
--     SELECT
--     film_id
--     FROM
--     inventory)
    
-- Exercise 8:

-- SELECT
--     *
--     FROM
--     payment
--     WHERE
--     amount BETWEEN 4 AND 6 
--     ORDER BY payment_date DESC

-- Exercise 9:

-- SELECT
--     first_name,
--     last_name
--     FROM
--         customer
--     ORDER BY first_name DESC
--     LIMIT 5
    
-- Exercise 10:

-- SELECT
--     first_name,
--     last_name
-- FROM
--     customer
-- ORDER BY first_name ASC
-- LIMIT 5
-- OFFSET 10

-- Exercise 11:

-- INSERT
--     INTO
--         customer(
--                 store_id,
--                 first_name,
--                 last_name,
--                 email,
--                 address_id
--                 )
--     VALUES
--         (1,
--         'Andrea',
--         'Viviani',
--         'andrea.viviani89@gmail.com',
--         1),
        
--         (1,
--         'test1',
--         'tes',
--         'tete@kk.it',
--         1),
        
--         (1,
--         'test2',
--         'te',
--         'tete@kk.it',
--         1),
        
--         (1,
--         'test3',
--         'tes',
--         'tete@kk.it',
--         1),
        
--         (1,
--         'test4',
--         'te',
--         'tete@kk.it',
--         1)
        
--         RETURNING *
        
-- Exercise 12:

-- UPDATE
--     customer
-- SET
--     first_name = 'yelp'
-- WHERE
--     customer_id = 606
-- RETURNING *
                
-- Exercise 13:

DELETE FROM
    customer
WHERE
    customer_id = 609
RETURNING *



    
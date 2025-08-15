-- init.sql

CREATE DATABASE IF NOT EXISTS mlops_db;
USE mlops_db;

-- BTC 데이터 테이블
CREATE TABLE IF NOT EXISTS btc_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    datetime DATETIME NOT NULL UNIQUE,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL
);

-- Gold 데이터 테이블
CREATE TABLE IF NOT EXISTS gold_data (
    id INT AUTO_INCREMENT PRIMARY KEY ,
    datetime DATETIME NOT NULL UNIQUE,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL
);

-- vix 데이터 테이블
CREATE TABLE IF NOT EXISTS vix_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    datetime DATETIME NOT NULL UNIQUE,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL
);

-- 나스닥 데이터 테이블
CREATE TABLE IF NOT EXISTS ndx100_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    datetime DATETIME NOT NULL UNIQUE,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL
);

CREATE TABLE IF NOT EXISTS integrated_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    datetime DATETIME NOT NULL UNIQUE,
    btc_open FLOAT NOT NULL,
    btc_high FLOAT NOT NULL,
    btc_low FLOAT NOT NULL,
    btc_close FLOAT NOT NULL,
    btc_volume FLOAT,
    ndx_open DECIMAL(20,8),
    ndx_high DECIMAL(20,8),
    ndx_low DECIMAL(20,8),
    ndx_close DECIMAL(20,8),
    vix_open DECIMAL(20,8),
    vix_high DECIMAL(20,8),
    vix_low DECIMAL(20,8),
    vix_close DECIMAL(20,8),
    gold_open DECIMAL(20,8),
    gold_high DECIMAL(20,8),
    gold_low DECIMAL(20,8),
    gold_close DECIMAL(20,8),
    gold_volume DECIMAL(20,8)
);



<<<<<<< HEAD
=======
-- 사용자 계정 생성 및 권한 부여
-- CREATE USER 'mlops_user'@'%' IDENTIFIED BY '0000';
CREATE USER IF NOT EXISTS 'mlops_user'@'%' IDENTIFIED BY '0000';
GRANT ALL PRIVILEGES ON mlops_db.* TO 'mlops_user'@'%';
>>>>>>> 8c31a5eb1f5f22b03063e25601884595b65a2016

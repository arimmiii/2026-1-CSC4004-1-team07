-- DB가 없으면 먼저 생성
CREATE DATABASE IF NOT EXISTS news_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- DB 선택
USE news_db;

CREATE TABLE ARTICLE (
    idx INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    link VARCHAR(2000) NOT NULL,
    category ENUM('정치', '경제', '사회', '생활/문화', 'IT/과학', '엔터', '스포츠') NOT NULL,
    category_detail JSON NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE USER (
    idx INT AUTO_INCREMENT PRIMARY KEY,
    id VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE USER_ARTICLE_ACTION (
    idx INT AUTO_INCREMENT PRIMARY KEY,
    user_idx INT NOT NULL,
    article_idx INT NOT NULL,
    action_type ENUM('시청', '좋아요') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_idx) REFERENCES USER(idx) ON DELETE CASCADE,
    FOREIGN KEY (article_idx) REFERENCES ARTICLE(idx) ON DELETE CASCADE,
    UNIQUE KEY uq_user_article_action (user_idx, article_idx, action_type)
);

ALTER TABLE ARTICLE ADD COLUMN content TEXT NULL;
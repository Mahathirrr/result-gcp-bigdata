#!/usr/bin/env python3
"""
Wikipedia Indonesia Topic Modeling dengan Hadoop + Spark
Menggunakan PySpark MLlib untuk distributed processing
"""

import sys
import os
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, length, size
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG = {
    'hdfs_path': 'hdfs://hadoop-master:9000/input/wikipedia_id_dataset.csv',
    'output_dir': '/app/output',
    'min_text_length': 100,
    'max_text_length': 50000,
    'min_tokens': 15,
    'num_topics': 20,
    'max_iterations': 25,
    'vocab_size': 25000,
    'min_df': 5,
    'max_df': 0.85,
    'terms_per_topic': 15,
    'seed': 42
}

class WikipediaTopicModelingHadoopSpark:
    """Topic Modeling menggunakan Hadoop + Spark dengan visualisasi lengkap"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spark = None
        self.stopwords_list = []
        logger.info("Wikipedia Topic Modeling dengan Hadoop + Spark")
        logger.info(f"Konfigurasi: {json.dumps(config, indent=2)}")

    def setup_spark(self) -> SparkSession:
        """Inisialisasi SparkSession untuk cluster"""
        logger.info("Menginisialisasi Spark Session...")

        spark = SparkSession.builder \
            .appName("Wikipedia Topic Modeling") \
            .master("spark://hadoop-master:7077") \
            .config("spark.executor.memory", "3g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.default.parallelism", "8") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        logger.info(f"Spark Version: {spark.version}")
        logger.info(f"Spark Master: {spark.sparkContext.master}")
        logger.info(f"App Name: {spark.sparkContext.appName}")

        self.spark = spark
        return spark

    def setup_stopwords(self) -> List[str]:
        """Setup stopwords dari NLTK untuk Bahasa Indonesia dan Inggris"""
        logger.info("Setup stopwords dari NLTK...")

        import nltk
        from nltk.corpus import stopwords

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        stopwords_indo = stopwords.words('indonesian')
        stopwords_eng = stopwords.words('english')

        all_stopwords = list(set(stopwords_indo + stopwords_eng))
        self.stopwords_list = all_stopwords

        logger.info(f"Stopwords Indonesia: {len(stopwords_indo)}")
        logger.info(f"Stopwords English: {len(stopwords_eng)}")
        logger.info(f"Total stopwords: {len(self.stopwords_list)}")

        return all_stopwords

    def load_data(self):
        """Load data dari HDFS"""
        logger.info(f"Loading data dari HDFS: {self.config['hdfs_path']}")

        df = self.spark.read.csv(
            self.config['hdfs_path'],
            header=True,
            inferSchema=True,
            mode='PERMISSIVE'
        )

        initial_count = df.count()
        logger.info(f"Total records dari HDFS: {initial_count:,}")

        df_filtered = df.filter(
            (col('text').isNotNull()) &
            (length(col('text')) >= self.config['min_text_length']) &
            (length(col('text')) <= self.config['max_text_length'])
        )

        filtered_count = df_filtered.count()
        retention_rate = (filtered_count / initial_count * 100) if initial_count > 0 else 0

        logger.info(f"Records setelah filter panjang: {filtered_count:,}")
        logger.info(f"Retention rate: {retention_rate:.2f}%")
        logger.info(f"Dropped: {initial_count - filtered_count:,} records")

        return df_filtered

    def preprocess_text(self, df):
        """Preprocessing text dengan Spark UDF"""
        logger.info("Memulai text preprocessing...")

        stopwords_set = set(self.stopwords_list)

        def tokenize_and_clean(text):
            if not text or not isinstance(text, str):
                return []

            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()

            tokens = [
                t for t in text.split()
                if 3 <= len(t) <= 20 and t not in stopwords_set
            ]

            return tokens

        tokenize_udf = udf(tokenize_and_clean, ArrayType(StringType()))

        df_processed = df.withColumn('tokens', tokenize_udf(col('text')))
        df_processed = df_processed.filter(size(col('tokens')) >= self.config['min_tokens'])

        processed_count = df_processed.count()
        logger.info(f"Records setelah preprocessing: {processed_count:,}")

        df_processed.cache()

        return df_processed

    def prepare_features(self, df):
        """Prepare features dengan CountVectorizer dan TF-IDF"""
        logger.info("Preparing features dengan CountVectorizer + TF-IDF...")

        cv_params = {
            'inputCol': 'tokens',
            'outputCol': 'raw_features',
            'minDF': self.config['min_df']
        }

        if self.config['vocab_size'] is not None:
            cv_params['vocabSize'] = self.config['vocab_size']

        cv = CountVectorizer(**cv_params)

        cv_model = cv.fit(df)
        df_cv = cv_model.transform(df)

        vocab = cv_model.vocabulary
        logger.info(f"Vocabulary size: {len(vocab)}")

        idf = IDF(inputCol='raw_features', outputCol='features')
        idf_model = idf.fit(df_cv)
        df_tfidf = idf_model.transform(df_cv)

        logger.info("Feature preparation selesai")

        return df_tfidf, vocab

    def train_lda(self, df):
        """Training LDA model dengan best practices"""
        logger.info("Memulai training LDA model...")
        logger.info(f"Jumlah topics: {self.config['num_topics']}")
        logger.info(f"Max iterations: {self.config['max_iterations']}")

        start_time = time.time()

        lda = LDA(
            k=self.config['num_topics'],
            maxIter=self.config['max_iterations'],
            seed=self.config['seed'],
            optimizer='online',
            learningDecay=0.51,
            learningOffset=1024.0,
            optimizeDocConcentration=True,
            subsamplingRate=0.05
        )

        logger.info("Training dimulai...")
        model = lda.fit(df)

        training_time = time.time() - start_time
        logger.info(f"Training selesai dalam {training_time:.2f} detik")

        log_likelihood = model.logLikelihood(df)
        log_perplexity = model.logPerplexity(df)
        logger.info(f"Log Likelihood: {log_likelihood:.2f}")
        logger.info(f"Log Perplexity: {log_perplexity:.2f}")

        return model, {
            'training_time': training_time,
            'log_likelihood': log_likelihood,
            'log_perplexity': log_perplexity
        }

    def extract_topics(self, model, vocab):
        """Extract top terms untuk setiap topic"""
        logger.info("Extracting topics...")

        topics = []
        topics_desc = model.describeTopics(maxTermsPerTopic=self.config['terms_per_topic'])

        for row in topics_desc.collect():
            topic_id = row['topic']
            term_indices = row['termIndices']
            term_weights = row['termWeights']

            top_terms = [
                (vocab[idx], float(weight))
                for idx, weight in zip(term_indices, term_weights)
            ]

            topics.append({
                'topic_id': topic_id,
                'top_terms': top_terms
            })

        logger.info(f"Extracted {len(topics)} topics")
        return topics

    def save_model(self, model, vocab):
        """Simpan LDA model dan vocabulary"""
        logger.info("Menyimpan LDA model...")

        output_dir = self.config['output_dir']
        model_path = f'{output_dir}/lda_model'

        try:
            # Gunakan overwrite untuk menghindari error jika folder sudah ada
            model.write().overwrite().save(model_path)

            # Verifikasi bahwa model benar-benar tersimpan
            import os
            if os.path.exists(model_path):
                logger.info(f"Model berhasil disimpan: {model_path}")
                # List isi folder model
                model_files = os.listdir(model_path)
                logger.info(f"Model files: {len(model_files)} files")
            else:
                logger.warning(f"Model path tidak ditemukan setelah save: {model_path}")
        except Exception as e:
            logger.error(f"Gagal menyimpan model: {e}")
            logger.warning("Model tidak tersimpan, melanjutkan tanpa model...")

        vocab_path = f'{output_dir}/vocabulary.json'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary disimpan: {vocab_path}")

    def save_results(self, topics, metrics, vocab_size, num_docs):
        """Simpan hasil ke file"""
        logger.info("Menyimpan hasil...")

        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        results_json = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_topics': self.config['num_topics'],
                'max_iterations': self.config['max_iterations'],
                'vocab_size': vocab_size,
                'num_docs': num_docs,
                'stopwords_count': len(self.stopwords_list),
                'training_time': metrics['training_time'],
                'log_likelihood': metrics.get('log_likelihood'),
                'log_perplexity': metrics.get('log_perplexity'),
                'config': self.config
            },
            'topics': topics
        }

        with open(f'{output_dir}/topics.json', 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        logger.info("Saved: topics.json")

        topics_csv_data = []
        for topic in topics:
            for rank, (term, weight) in enumerate(topic['top_terms'], 1):
                topics_csv_data.append({
                    'topic_id': topic['topic_id'],
                    'rank': rank,
                    'term': term,
                    'weight': weight
                })

        pd.DataFrame(topics_csv_data).to_csv(f'{output_dir}/topics.csv', index=False)
        logger.info("Saved: topics.csv")

        with open(f'{output_dir}/metrics.json', 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': vocab_size,
                'num_docs': num_docs,
                'num_topics': self.config['num_topics'],
                'stopwords_count': len(self.stopwords_list),
                **metrics,
                'config': self.config
            }, f, indent=2)
        logger.info("Saved: metrics.json")

    def create_visualizations(self, topics, model, df_features):
        """Generate semua visualisasi"""
        logger.info("Generating visualizations...")

        output_dir = self.config['output_dir']

        self._create_topic_terms_bars(topics, output_dir)
        self._create_wordclouds(topics, output_dir)
        self._create_topic_dominance_chart(model, df_features, output_dir)
        self._create_document_topic_heatmap(model, df_features, output_dir)

        logger.info("Semua visualisasi selesai dibuat")

    def _create_topic_terms_bars(self, topics, output_dir):
        """Bar chart untuk top terms setiap topic"""
        logger.info("Creating bar charts untuk top terms...")

        num_topics = len(topics)
        fig, axes = plt.subplots(num_topics, 1, figsize=(14, 4 * num_topics))

        if num_topics == 1:
            axes = [axes]

        for idx, topic_data in enumerate(topics):
            terms = [t[0] for t in topic_data['top_terms']]
            weights = [t[1] for t in topic_data['top_terms']]

            ax = axes[idx]
            bars = ax.barh(range(len(terms)), weights, color=plt.cm.viridis(idx / num_topics))
            ax.set_yticks(range(len(terms)))
            ax.set_yticklabels(terms, fontsize=10)
            ax.set_xlabel('Weight', fontsize=11, fontweight='bold')
            ax.set_title(f'Topic {idx}: Top {len(terms)} Terms',
                        fontsize=12, fontweight='bold', pad=10)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            for i, (bar, weight) in enumerate(zip(bars, weights)):
                ax.text(weight, i, f' {weight:.3f}',
                       va='center', fontsize=8, color='black')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/topics_top_terms.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: topics_top_terms.png")
        plt.close()

    def _create_wordclouds(self, topics, output_dir):
        """Word clouds untuk setiap topic"""
        logger.info("Creating word clouds...")

        num_topics = len(topics)
        num_cols = min(4, num_topics)
        num_rows = (num_topics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

        if num_topics == 1:
            axes = np.array([axes])

        axes_flat = axes.flatten() if num_topics > 1 else [axes]

        for idx, topic_data in enumerate(topics):
            word_freq = dict(topic_data['top_terms'])

            wc = WordCloud(
                width=500,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(word_freq)

            ax = axes_flat[idx]
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(f'Topic {idx}', fontsize=14, fontweight='bold', pad=10)
            ax.axis('off')

        for idx in range(num_topics, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.suptitle('Topic Word Clouds', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/topics_wordclouds.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: topics_wordclouds.png")
        plt.close()

    def _create_topic_dominance_chart(self, model, df_features, output_dir):
        """Bar chart untuk dominance setiap topic"""
        logger.info("Creating topic dominance chart...")

        def get_dominant_topic(vec):
            return int(np.argmax(vec.toArray()))

        dominant_udf = udf(get_dominant_topic, IntegerType())

        transformed = model.transform(df_features)
        with_dominant = transformed.withColumn('dominant_topic',
                                              dominant_udf(col('topicDistribution')))

        topic_counts = with_dominant.groupBy('dominant_topic').count().collect()
        topic_counts_sorted = sorted(topic_counts, key=lambda x: x['dominant_topic'])

        topics_list = [row['dominant_topic'] for row in topic_counts_sorted]
        counts = [row['count'] for row in topic_counts_sorted]

        plt.figure(figsize=(14, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(topics_list)))
        bars = plt.bar(topics_list, counts, color=colors, edgecolor='black', linewidth=0.7)

        plt.xlabel('Topic ID', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Documents', fontsize=12, fontweight='bold')
        plt.title('Topic Dominance - Document Distribution across Topics',
                 fontsize=14, fontweight='bold', pad=15)
        plt.xticks(topics_list)
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/topic_dominance.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: topic_dominance.png")
        plt.close()

    def _create_document_topic_heatmap(self, model, df_features, output_dir):
        """Heatmap untuk distribusi topic pada sample dokumen"""
        logger.info("Creating document-topic distribution heatmap...")

        transformed = model.transform(df_features)
        sample = transformed.select('topicDistribution').limit(20).collect()

        dist_matrix = np.array([row['topicDistribution'].toArray() for row in sample])

        plt.figure(figsize=(12, 8))

        sns.heatmap(
            dist_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=[f'T{i}' for i in range(self.config['num_topics'])],
            yticklabels=[f'D{i}' for i in range(len(sample))],
            cbar_kws={'label': 'Topic Probability', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='white'
        )

        plt.xlabel('Topic ID', fontsize=12, fontweight='bold')
        plt.ylabel('Document ID (Sample)', fontsize=12, fontweight='bold')
        plt.title('Document-Topic Distribution Heatmap\n(Sample of 20 Documents)',
                 fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/document_topic_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: document_topic_distribution.png")
        plt.close()

    def run_pipeline(self):
        """Jalankan pipeline lengkap"""
        logger.info("MEMULAI PIPELINE")
        start_time = datetime.now()

        try:
            self.setup_spark()
            self.setup_stopwords()

            df = self.load_data()
            df_processed = self.preprocess_text(df)
            df_features, vocab = self.prepare_features(df_processed)

            num_docs = df_features.count()
            vocab_size = len(vocab)

            logger.info(f"Final dataset: {num_docs:,} documents")
            logger.info(f"Vocabulary size: {vocab_size:,} words")

            model, metrics = self.train_lda(df_features)
            topics = self.extract_topics(model, vocab)

            self.save_model(model, vocab)
            self.save_results(topics, metrics, vocab_size, num_docs)
            self.create_visualizations(topics, model, df_features)

            duration = (datetime.now() - start_time).total_seconds() / 60
            logger.info("PIPELINE SELESAI")
            logger.info(f"Durasi: {duration:.2f} menit")
            logger.info(f"Topics: {len(topics)}")
            logger.info(f"Documents: {num_docs:,}")
            logger.info(f"Vocabulary: {vocab_size:,}")
            logger.info(f"Output: {self.config['output_dir']}/")

        except Exception as e:
            logger.error(f"Pipeline gagal: {e}", exc_info=True)
            raise
        finally:
            if self.spark:
                self.spark.stop()
                logger.info("Spark session dihentikan")

def main():
    """Main function"""
    modeler = WikipediaTopicModelingHadoopSpark(CONFIG)
    modeler.run_pipeline()

if __name__ == '__main__':
    main()
import unittest
from io import BytesIO

from app import app


class AppTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_pages_and_health(self):
        for path in ('/', '/about', '/results'):
            self.assertEqual(self.client.get(path).status_code, 200)
        health = self.client.get('/api/health').get_json()
        self.assertEqual(health['status'], 'healthy')
        self.assertIn(health['engine'], {'heuristic', 'distilbert-multimodal'})
        self.assertEqual(health['model_loaded'], health['engine'] == 'distilbert-multimodal')

        index_html = self.client.get('/').get_data(as_text=True)
        for element_id in ('resultPrediction', 'resultConfidence', 'resultEngine', 'probabilityValue', 'timestamp'):
            self.assertIn(f'id="{element_id}"', index_html)
        self.assertIn('script.js?v=4', index_html)

    def test_samples_can_be_analyzed(self):
        legitimate = self.client.get('/sample/legitimate').get_json()
        phishing = self.client.get('/sample/phishing').get_json()

        legitimate_result = self.client.post('/analyze', json=legitimate)
        phishing_result = self.client.post('/analyze', json=phishing)

        self.assertEqual(legitimate_result.status_code, 200)
        self.assertEqual(phishing_result.status_code, 200)
        self.assertEqual(legitimate_result.get_json()['prediction'], 'Legitimate')
        self.assertEqual(phishing_result.get_json()['prediction'], 'Phishing')
        contributions = phishing_result.get_json()['feature_contributions']
        self.assertTrue(contributions)
        self.assertTrue(all(item['contribution'] >= 0 for item in contributions))
        if phishing_result.get_json()['engine'] == 'distilbert-multimodal':
            self.assertTrue(phishing_result.get_json()['text_explanations'])

    def test_invalid_requests_return_400(self):
        self.assertEqual(self.client.post('/analyze').status_code, 400)
        self.assertEqual(self.client.post('/analyze', json={}).status_code, 400)
        self.assertEqual(
            self.client.post('/analyze', json={'sender': 1, 'subject': 'x', 'body': 'y'}).status_code,
            400,
        )

    def test_suspicious_tld_is_detected_with_a_path(self):
        response = self.client.post('/analyze', json={
            'sender': 'person@example.com',
            'subject': 'A message',
            'body': 'Please review this message.',
            'urls': 'https://example.xyz/account/verify?token=1',
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['features']['has_suspicious_tld'], 1)

    def test_eml_upload_extracts_fields_and_urls(self):
        message = (
            b'From: Security <security@example.com>\r\n'
            b'Subject: Verify your account\r\n'
            b'Content-Type: text/plain; charset=utf-8\r\n\r\n'
            b'Visit https://example.xyz/account now.'
        )
        response = self.client.post(
            '/upload',
            data={'email_file': (BytesIO(message), 'message.eml')},
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 200)
        parsed = response.get_json()
        self.assertEqual(parsed['subject'], 'Verify your account')
        self.assertEqual(parsed['sender'], 'security@example.com')
        self.assertEqual(parsed['urls'], 'https://example.xyz/account')

    def test_upload_rejects_unsupported_files(self):
        response = self.client.post(
            '/upload',
            data={'email_file': (BytesIO(b'data'), 'message.pdf')},
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 400)

    def test_body_urls_are_analyzed_without_manual_url_field(self):
        response = self.client.post('/analyze', json={
            'sender': 'person@example.com',
            'subject': 'Please review',
            'body': 'Open https://login.example.xyz/account immediately.',
            'urls': '',
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result['features']['url_count'], 1)
        self.assertEqual(result['features']['has_suspicious_tld'], 1)


if __name__ == '__main__':
    unittest.main()

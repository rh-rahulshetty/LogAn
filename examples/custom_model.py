from gliner2 import GLiNER2
from log_diagnosis.models.manager import ModelTemplate  # Binded dynamically during runtime import


class GLiNERModel(ModelTemplate):
    def init_model(self):
        """Initialize the model. Called once after construction."""
        self.extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    def classify_golden_signal(self, input: list[str], batch_size: int = 32) -> list[dict]:
        '''
        Returns
        [
            {
                'labels': ['label1'],
                'scores': [0.1]
            }
        ]
        '''
        schema = self.extractor.create_schema().classification(
            "golden_signal",
            {
                "information": "Classifies log lines that provide general operational details, status updates, or other non-critical messages.",
                "error": "Classifies log lines that indicate an error condition, malfunction, or abnormal system event.",
                "availability": "Classifies log lines related to system uptime, downtime, service availability, or accessibility.",
                "latency": "Classifies log lines that refer to delays, response times, or performance lag within the system.",
                "saturation": "Classifies log lines that signify resource exhaustion, capacity issues, or overloaded subsystems.",
                "traffic": "Classifies log lines describing system throughput, number of requests, connections, or data flow.",
            }
        )

        results = self.extractor.batch_extract(
            input,
            schema,
            batch_size=batch_size,
            threshold=0.5,
            include_confidence=True,
        )

        formatted_result = []
        for result in results:
            classification_result = result['golden_signal']
            formatted_result.append({
                'labels': [classification_result['label']],
                'scores': [classification_result['confidence']]
            })
        return formatted_result

    def classify_fault_category(self, input: list[str], batch_size: int = 32) -> list[dict]:
        '''
        Returns list of dictionaries with 'labels' and 'scores' keys.
        [
            {
                'labels': ['label1', 'label2', 'label3'],
                'scores': [0.1, 0.2, 0.3]
            }
        ]
        '''
        schema = self.extractor.create_schema().classification(
            "fault_category",
            {
                "io": "Classifies log lines that refer to input/output operations, file system interactions, or data transfer.",
                "authentication": "Classifies log lines that relate to user authentication, authorization, or security protocols.",
                "network": "Classifies log lines that describe network connectivity, routing, or protocol interactions.",
                "application": "Classifies log lines that refer to application-level events, process interactions, or service calls.",
                "device": "Classifies log lines that relate to hardware device status, driver activity, or peripheral operations.",
            },
            multi_label=True,
            cls_threshold=0.3,
        )

        results = self.extractor.batch_extract(
            input,
            schema,
            batch_size=batch_size,
            threshold=0.3,
            include_confidence=True,
        )

        formatted_result = []
        for result in results:
            classification_results = result['relation_extraction']['fault_category']
            label_score = {
                'labels': [],
                'scores': []
            }
            for label, score in classification_results:
                label_score['labels'].append(label)
                label_score['scores'].append(score)
            formatted_result.append(label_score)
        return formatted_result

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token


class AdaptiveKGMetaModel(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        return self.get_model().mm_projector(image_features), image_features


class AdaptiveKGConstructionRL(AdaptiveKGMetaModel, nn.Module):
    def __init__(self, config, action_dim, gamma=0.99, lr=1e-4):
        super(AdaptiveKGConstructionRL, self).__init__()

        # Initialize policy and critic networks
        self.actor = PolicyNetwork(action_dim)
        self.critic = CriticNetwork()
        self.gamma = gamma
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        # Config and output directory for checkpoints
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("lora-trained-llm")
        self.model, self.image_processor = self._load_latest_checkpoint_model()

    def _load_latest_checkpoint_model(self):
        """
        Load the latest model checkpoint from output_dir specified in config.
        """
        output_dir = self.config.output_dir
        checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.bin')]
        latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
        model_path = os.path.join(output_dir, latest_checkpoint)

        base_model_path = self.config.base_model_path  # specify base model directory
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, base_model_path, latest_checkpoint)
        return model, image_processor

    def select_triples(self, triples, image):
        """
        Selects a subset of triples iteratively to form the subgraph using reinforcement learning.
        Parameters:
            triples: List of knowledge graph triples
            image: The image tensor used as context
        Returns:
            selected_triples: Final adaptive knowledge subgraph
        """
        image_features, _ = self.encode_images(image)
        selected_triples = []

        # Initialize graph state
        subgraph_entities = set()
        subgraph_relations = set()

        for _ in range(len(triples)):  # Iterate over the triples for selecting a subset
            # Select actions based on current policy
            action_probs = self.actor(image_features)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            selected_triple = triples[action.item()]

            # Update the subgraph with selected triple
            entity, relation = selected_triple[0], selected_triple[1]
            if entity not in subgraph_entities:
                subgraph_entities.add(entity)
                subgraph_relations.add(relation)
                selected_triples.append(selected_triple)

            # Stop if subgraph meets certain size (e.g., top 10 triples)
            if len(selected_triples) >= 10:
                break

        return selected_triples

    def compute_rewards(self, visual_features, entity_features):
        image_relevance = F.cosine_similarity(visual_features, entity_features, dim=1).mean()
        return image_relevance

    def terminal_reward(self, image_feature, distractors, generated_caption):
        """
        Final reward calculation based on the generated captions from the latest checkpoint model.
        """
        # Generate caption with the latest checkpoint model
        caption_features = self.evaluate_model_for_reward(generated_caption, image_feature)

        target_score = F.cosine_similarity(caption_features, image_feature, dim=1)
        distractor_scores = F.cosine_similarity(caption_features.unsqueeze(0), distractors, dim=1)
        terminal_reward = torch.log(target_score / (distractor_scores.sum() + 1e-6))
        return terminal_reward

    def evaluate_model_for_reward(self, generated_caption, image_tensor):
        """
        Generate caption using the latest model checkpoint for evaluation and reward calculation.
        """
        # Tokenize input caption
        input_ids = self.tokenizer(generated_caption, return_tensors="pt").input_ids.to(device="cuda")

        # Generate output with model
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda"),
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                max_new_tokens=100,
                use_cache=True
            )

        # Decode and tokenize the generated caption for further comparison
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_features = self.tokenizer(output_text, return_tensors="pt").input_ids.to(device="cuda")
        return output_features

    def update_policy(self, rewards, log_probs, values):
        advantages = rewards - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, rewards.detach())
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def train_step(self, visual_features, entity_features, generated_caption, image_feature, distractors, triples,
                   image):
        selected_triples = self.select_triples(triples, image)
        value = self.critic(image_feature)

        # Calculate rewards
        image_relevance_reward = self.compute_rewards(visual_features, entity_features)
        terminal_reward = self.terminal_reward(image_feature, distractors, generated_caption)

        total_reward = 0.5 * image_relevance_reward + 0.5 * terminal_reward
        self.update_policy(total_reward, value)
        return selected_triples, total_reward


class PolicyNetwork(nn.Module):
    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = self.value_head(x)
        return value

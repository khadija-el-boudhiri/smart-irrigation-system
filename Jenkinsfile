pipeline {
  agent {
    docker {
      image 'docker:27-dind'
      args '--privileged -v /var/run/docker.sock:/var/run/docker.sock'
    }
  }

  environment {
    MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
    MODEL_NAME          = 'PlantWaterModel'
  }

  parameters {
    choice(
      name: 'DEPLOY_ENV',
      choices: ['staging', 'production'],
      description: 'Target deployment environment'
    )
    booleanParam(
      name: 'SKIP_TRAIN',
      defaultValue: false,
      description: 'Skip DVC pull, training, and evaluation'
    )
  }

  stages {

    stage('Checkout') {
      steps {
        checkout([
          $class: 'GitSCM',
          branches: scm.branches,
          extensions: [[$class: 'CloneOption', shallow: true, depth: 1]],
          userRemoteConfigs: scm.userRemoteConfigs
        ])
      }
    }

    stage('Lint & Test') {
      agent { docker { image 'python:3.11-slim' } }
      steps {
        sh 'pip install --quiet flake8 pytest'
        sh 'flake8 src/ --max-line-length=120'
        sh 'pytest tests/ -v'
      }
    }

    stage('DVC Pull') {
      when { expression { !params.SKIP_TRAIN } }
      steps {
        withCredentials([string(credentialsId: 'DVC_ACCESS_KEY',
                                variable: 'DVC_ACCESS_KEY')]) {
          sh 'dvc pull'
        }
      }
    }

    stage('Train') {
      when { expression { !params.SKIP_TRAIN } }
      steps {
        sh 'python src/train_models.py'
      }
    }

    stage('Evaluate') {
      when { expression { !params.SKIP_TRAIN } }
      steps {
        sh 'python src/promote_model.py'
      }
    }

    stage('Build Docker Image') {
      steps {
        sh 'docker build -t smart-irrigation/api:${BUILD_NUMBER} -f Dockerfile.api .'
        withCredentials([usernamePassword(
          credentialsId: 'DOCKER_REGISTRY_CREDENTIALS',
          usernameVariable: 'DOCKER_USER',
          passwordVariable: 'DOCKER_PASS'
        )]) {
          sh 'echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin'
          sh 'docker push smart-irrigation/api:${BUILD_NUMBER}'
          sh 'docker tag smart-irrigation/api:${BUILD_NUMBER} smart-irrigation/api:latest'
          sh 'docker push smart-irrigation/api:latest'
        }
      }
    }

    stage('Deploy') {
      steps {
        sh 'docker compose up -d --no-deps api'
      }
    }

    stage('Post') {
      steps {
        echo "Deployed smart-irrigation/api:${BUILD_NUMBER} to ${params.DEPLOY_ENV}"
      }
    }

  }

  post {
    always {
      archiveArtifacts artifacts: 'mlruns/**', allowEmptyArchive: true
    }
    failure {
      withCredentials([string(credentialsId: 'SLACK_WEBHOOK',
                              variable: 'SLACK_WEBHOOK')]) {
        sh '''
          curl -s -X POST -H "Content-type: application/json" \
          --data "{\\"text\\":\\"Build failed: ${JOB_NAME} #${BUILD_NUMBER} — ${BUILD_URL}\\"}" \
          "$SLACK_WEBHOOK"
        '''
      }
    }
  }
}

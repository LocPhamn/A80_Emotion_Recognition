const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const Statistic = sequelize.define('Statistic', {
    idstatistic: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true,
      field: 'idstatistic'
    },
    video_id: {
      type: DataTypes.INTEGER,
      allowNull: true,
      references: {
        model: 'video',
        key: 'idvideo'
      }
    },
    total_visitor: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    happy_rate: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    neutral_rate: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    disgust_rate: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    surprise_rate: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    fear_rate: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    angry_rate: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    sad_rate: {
      type: DataTypes.FLOAT,
      allowNull: true
    }
  }, {
    tableName: 'statistic',
    timestamps: false
  });

  Statistic.associate = (models) => {
    // Statistic belongs to Video
    Statistic.belongsTo(models.Video, {
      foreignKey: 'video_id',
      as: 'video'
    });
  };

  return Statistic;
};
